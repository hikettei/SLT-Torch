
#Ref: https://github.com/joennlae/halutmatmul

import copy
import numba
from numba import prange
import numpy as np

from sklearn import linear_model
import os, resource
from typing import Any, List, Optional, Tuple, Union, Literal, TYPE_CHECKING
from .cffi_utils import (
    maddness_encode_cpp,
    maddness_scan
    )

@numba.njit(fastmath=True, cache=True, parallel=False)
def _cumsse_cols(X):
    # TODO: can be optimized with numpy
    N, D = X.shape
    cumsses = np.empty((N, D), X.dtype)
    cumX_column = np.empty(D, X.dtype)
    cumX2_column = np.empty(D, X.dtype)
    for j in range(D):
        cumX_column[j] = X[0, j]
        cumX2_column[j] = X[0, j] * X[0, j]
        cumsses[0, j] = 0  # no err in bucket with 1 element
    for i in range(1, N):
        one_over_count = 1.0 / (i + 1)
        for j in range(D):
            cumX_column[j] += X[i, j]
            cumX2_column[j] += X[i, j] * X[i, j]
            meanX = cumX_column[j] * one_over_count
            cumsses[i, j] = cumX2_column[j] - (cumX_column[j] * meanX)
    return cumsses


def optimal_split_val(
    X,
    dim,
    X_orig=None,
):

    X_orig = X if X_orig is None else X_orig
    if X_orig.shape != X.shape:
        assert X_orig.shape == X.shape

    N, _ = X.shape
    sort_idxs = np.argsort(X_orig[:, dim])
    X_sort = X[sort_idxs]

    # cumulative SSE (sum of squared errors)
    sses_head = _cumsse_cols(X_sort)
     
    sses_tail = _cumsse_cols(X_sort[::-1])[::-1]
    sses = sses_head
    sses[:-1] += sses_tail[1:]
    sses = sses.sum(axis=1)

    best_idx = np.argmin(sses)
    next_idx = min(N - 1, best_idx + 1)
    col = X[:, dim]
    best_val = (col[sort_idxs[best_idx]] + col[sort_idxs[next_idx]]) / 2

    return best_val, sses[best_idx]

class Bucket:
    """
    sumX and sumX2 are (IDxs_per_codebook, 1) e.g (512 // 16, 1)
    """

    __slots__ = "N D id sumX sumX2 point_ids support_add_and_remove".split()

    def __init__(
        self,
        D=None,
        N=0,
        sumX=None,
        sumX2=None,
        point_ids=None,
        bucket_id=0,
        support_add_and_remove=False,
    ):
        # self.reset(D=D, sumX=sumX, sumX2=sumX2)
        # assert point_ids is not None
        if point_ids is None:
            assert N == 0
            point_ids = (
                set() if support_add_and_remove else np.array([], dtype=np.int64)
            )

        self.N = len(point_ids)
        self.id = bucket_id
        # this is just so that we can store the point ids as array instead of
        # set, while still retaining option to run our old code that needs
        # them to be a set for efficient inserts and deletes
        self.support_add_and_remove = support_add_and_remove
        if support_add_and_remove:
            self.point_ids = set(point_ids)
        else:
            self.point_ids = np.asarray(point_ids)

        # figure out D
        if (D is None or D < 1) and (sumX is not None):
            D = len(sumX)
        elif (D is None or D < 1) and (sumX2 is not None):
            D = len(sumX2)
        assert D is not None
        self.D = D

        # figure out + sanity check stats arrays
        self.sumX = np.zeros(D, dtype=np.float32) if (sumX is None) else sumX
        self.sumX2 = np.zeros(D, dtype=np.float32) if (sumX2 is None) else sumX2  # noqa
        # print("D: ", D)
        # print("sumX type: ", type(sumX))
        assert len(self.sumX) == D
        assert len(self.sumX2) == D
        self.sumX = np.asarray(self.sumX).astype(np.float32)
        self.sumX2 = np.asarray(self.sumX2).astype(np.float32)

    def add_point(self, point, point_id=None):
        assert self.support_add_and_remove
        # TODO replace with more numerically stable updates if necessary
        self.N += 1
        self.sumX += point
        self.sumX2 += point * point
        if point_id is not None:
            self.point_ids.add(point_id)

    def remove_point(self, point, point_id=None):
        assert self.support_add_and_remove
        self.N -= 1
        self.sumX -= point
        self.sumX2 -= point * point
        if point_id is not None:
            self.point_ids.remove(point_id)

    def deepcopy(self, bucket_id=None):  # deep copy
        bucket_id = self.id if bucket_id is None else bucket_id
        return Bucket(
            sumX=np.copy(self.sumX),
            sumX2=np.copy(self.sumX2),
            point_ids=copy.deepcopy(self.point_ids),
            bucket_id=bucket_id,
        )
        
    def split(self, X=None, dim=None, val=None, X_orig=None):
        id0 = 2 * self.id #left
        id1 = id0 + 1 #right
        if X is None or self.N < 2:  # copy of this bucket + an empty bucket
            return (self.deepcopy(bucket_id=id0), Bucket(D=self.D, bucket_id=id1))
        assert self.point_ids is not None
        my_idxs = np.asarray(self.point_ids)

        X = X_orig[my_idxs]
        X_orig = X if X_orig is None else X_orig[my_idxs]
        mask = X[:, dim] > val  #
        not_mask = ~mask
        X0 = X[not_mask]
        X1 = X[mask]
        ids0 = my_idxs[not_mask]
        ids1 = my_idxs[mask]

        def create_bucket(points, ids, bucket_id):
            sumX = points.sum(axis=0) if len(ids) else None
            sumX2 = (points * points).sum(axis=0) if len(ids) else None
            return Bucket(
                D=self.D, point_ids=ids, sumX=sumX, sumX2=sumX2, bucket_id=bucket_id
            )

        return create_bucket(X0, ids0, id0), create_bucket(X1, ids1, id1)

    def optimal_split_val(self, X, dim, X_orig=None):
        if self.N < 2 or self.point_ids is None:
            return 0, 0
        my_idxs = np.asarray(self.point_ids)
        if X_orig is not None:
            X_orig = X_orig[my_idxs]
        return optimal_split_val(X[my_idxs], dim, X_orig=X_orig)
        # return optimal_split_val_new(X, dim, my_idxs)

    def col_means(self):
        return self.sumX.astype(np.float64) / max(1, self.N)

    def col_variances(self, safe=False):
        if self.N < 1:

            return np.zeros(self.D, dtype=np.float32)
        E_X2 = self.sumX2 / self.N
        E_X = self.sumX / self.N
        ret = E_X2 - (E_X * E_X)
        return np.maximum(0, ret) if safe else ret

    def col_sum_sqs(self):
        return self.col_variances() * self.N

    @property
    def loss(self):
        # more stable version, that also clamps variance at 0
        return max(0, np.sum(self.col_sum_sqs()))

def create_codebook_start_end_idxs(X, number_of_codebooks, algo="start"):
    """
    returns vector (C, 2)
    [
      start_idx_0, end_idx_0,
      start_idx_1, end_idx_1,
      ...
    ]
    """
    assert algo in ("start", "end")

    _, D = X.shape
    number_of_codebooks = int(number_of_codebooks)
    assert D >= number_of_codebooks

    idxs = np.empty((number_of_codebooks, 2), dtype=np.int64)
    full_subvec_len = D // number_of_codebooks
    start_idx = 0
    for c in range(number_of_codebooks):
        subvec_len = full_subvec_len
        if algo == "start":  # wider codebooks at the start
            if c < (D % number_of_codebooks):
                subvec_len += 1
        elif algo == "end":  # wider codebooks at the end
            if (number_of_codebooks - c - 1) < (D % number_of_codebooks):
                subvec_len += 1
        end_idx = min(D, start_idx + subvec_len)
        idxs[c, 0] = start_idx
        idxs[c, 1] = end_idx

        start_idx = end_idx

    assert idxs[0, 0] == 0
    assert idxs[-1, -1] == D
    return idxs


class MultiSplit:
    __slots__ = "dim vals scaleby offset".split()

    def __init__(self, dim, vals, scaleby=None, offset=None):
        self.dim = dim
        self.vals = np.asarray(vals)
        self.scaleby = scaleby
        self.offset = offset

    def __repr__(self) -> str:
        return f"<{self.get_params()}>"

    def __str__(self) -> str:
        return self.get_params()

    def get_params(self) -> str:
        params = (
            f"Multisplit: dim({self.dim}), vals({self.vals}), "
            f"scaleby({self.scaleby}), offset({self.offset})"
        )
        return params

    def preprocess_x(self, x: np.ndarray) -> np.ndarray:
        if self.offset is not None:
            x = x - self.offset
        if self.scaleby is not None:
            x = x * self.scaleby
        return x


def learn_binary_tree_splits(
    X: np.ndarray,
    nsplits: int = 4,  # levels of resulting binary hash tree
    return_prototypes: bool = True,
    # return_buckets: bool = False,
    X_orig: Optional[np.ndarray] = None,
    check_x_dims: int = 8,  # can be used to check more or less dims with max losses
    learn_quantize_params: bool = False,
) -> Tuple[list, int, Union[list, np.ndarray]]:
    assert nsplits <= 4  # >4 splits means >16 split_vals for this func's impl

    X = X.copy().astype(np.float32)
    N, D = X.shape  # D amount of IDx per codebook
    X_orig = X.copy() if X_orig is None else X_orig.copy()

    # initially, one big bucket with everything
    buckets = [
        Bucket(sumX=X.sum(axis=0), sumX2=(X * X).sum(axis=0), point_ids=np.arange(N))
    ]
    # total_loss = sum([bucket.loss for bucket in buckets])

    # print("================================")
    # print("learn_binary_tree_splits(): initial loss:  ", total_loss)

    splits = []
    col_losses = np.zeros(D, dtype=np.float32)
    OFFSET = 0.0
    SCALE_BY = 1.0
    X = X * SCALE_BY + OFFSET
    # X_orig = X_orig + OFFSET
    for _ in range(nsplits):
        # in the original code there are more strategies: eigenvec, bucket_eigenvecs, kurtosis
        # dim_heuristic == "bucket_sse":
        col_losses[:] = 0  # set all zero
        for buck in buckets:
            col_losses += buck.col_sum_sqs()  # return variance
        try_dims = np.argsort(col_losses)[::-1][
            :check_x_dims
        ]  # choose biggest column losses
        losses = np.zeros(len(try_dims), dtype=X.dtype)
        all_split_vals = []  # vals chosen by each bucket/group for each dim

        # determine for this dim what the best split vals are for each
        # group and what the loss is when using these split vals
        for d, dim in enumerate(try_dims):
            split_vals = []  # each bucket contributes one split val
            for _, buck in enumerate(buckets):
                # X.shape (50000, 32), dim is a number 0-31, val 1D, loss 1D
                val, loss = buck.optimal_split_val(X, dim, X_orig=X_orig)
                losses[d] += loss
                if d > 0 and losses[d] >= np.min(losses[:d]):
                    # early stop
                    break
                split_vals.append(val)
            all_split_vals.append(split_vals)
        # determine best dim to split on, and pull out associated split
        # vals for all buckets

        best_tried_dim_idx = np.argmin(losses)
        best_dim = try_dims[best_tried_dim_idx]
        use_split_vals = all_split_vals[best_tried_dim_idx]
        split = MultiSplit(dim=best_dim, vals=use_split_vals)

        if learn_quantize_params:
            # simple version, which also handles 1 bucket: just set min
            # value to be avg of min splitval and xval, and max value to
            # be avg of max splitval and xval
            x = X[:, best_dim]  # Vector (50000, 1)
            offset = (np.min(x) + np.min(use_split_vals)) / 2
            upper_val = (np.max(x) + np.max(use_split_vals)) / 2 - offset
            # TODO: why this specific scale value??
            scale = 254.0 / upper_val
            # if learn_quantize_params == "int16":
            scale = 2.0 ** int(np.log2(scale))

            split.offset = offset
            split.scaleby = scale
            split.vals = (split.vals - split.offset) * split.scaleby
            # TODO: look at clippings
            split.vals = np.clip(split.vals, 0, 255).astype(np.int32)
        else:
            split.offset = OFFSET
            split.scaleby = SCALE_BY
        splits.append(split)

        # apply this split to get next round of buckets
        new_buckets = []
        for i, buck in enumerate(buckets):
            val = use_split_vals[i]
            new_buckets += list(buck.split(X, dim=best_dim, val=val, X_orig=X_orig))
        buckets = new_buckets

    # pylint: disable=consider-using-generator
    loss = sum([bucket.loss for bucket in buckets])
    # print("learn_binary_tree_splits(): returning loss: ", loss)

    if return_prototypes:
        prototypes = np.vstack([buck.col_means() for buck in buckets])
        assert prototypes.shape == (len(buckets), X.shape[1])
        return splits, loss, prototypes
    # if return_buckets:
    return splits, loss, buckets


def init_and_learn_hash_function(
    X: np.ndarray, C: int, pq_perm_algo: str = "start"
) -> Tuple[np.ndarray, list, np.ndarray, list]:
    _, D = X.shape
    K = 16

    X = X.astype(np.float32)
    X_error = X.copy().astype(np.float32)
    X_orig = X

    all_prototypes = np.zeros((C, K, D), dtype=np.float32)
    all_splits: List = []
    pq_idxs = create_codebook_start_end_idxs(X, C, algo=pq_perm_algo)

    # ------------------------ 0th iteration; initialize all codebooks
    all_splits = []
    all_buckets = []
    for c in range(C):
        start_idx, end_idx = pq_idxs[c]
        idxs = np.arange(start_idx, end_idx)
        # in original code there is other selections based on PCA and disjoint PCA

        use_X_error = X_error[:, idxs]
        use_X_orig = X_orig[:, idxs]

        # learn codebook to soak current residuals
        multisplits, _, buckets = learn_binary_tree_splits(
            use_X_error, X_orig=use_X_orig, return_prototypes=False
        )

        for split in multisplits:
            split.dim = idxs[split.dim]
        all_splits.append(multisplits)
        all_buckets.append(buckets)

        # update residuals and store prototypes
        # idxs = IDs that were look at for current codebook
        # buck.point_ids = rows that landed in certain K
        #   [    0     5    21 ... 99950 99979 99999] (N=100000)
        # X_error = is here still the A input
        # remove centroid from all the points that lie in a certain codebook
        # set prototype value
        centroid = np.zeros(D, dtype=np.float32)
        for b, buck in enumerate(buckets):
            # print(b, idxs, buck.point_ids, centroid, buck.col_means())
            if len(buck.point_ids):
                centroid[:] = 0
                centroid[idxs] = buck.col_means()
                X_error[buck.point_ids] -= centroid
                # update centroid here in case we want to regularize it somehow
                all_prototypes[c, b] = centroid

        # X_error = A_input - all_centroids
        ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(
            f"Learning progress {X.shape}-{C}-{K}: {c + 1}/{C} "
            f"({(ram_usage / (1024 * 1024)):.3f} GB)"
        )
    return X_error, all_splits, all_prototypes, all_buckets



def sparsify_and_int8_A_enc(A_enc, K=16):
    """
    returns X_binary from an encoded Matrix [N, C] vals (0-K)
    to
    [[0 0 0 ... 0 0 0]
     [0 0 1 ... 0 0 0]
     ...
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 1 0 0]
     [0 0 0 ... 0 0 0]]
    """
    N, C = A_enc.shape
    D = C * K
    out = np.zeros((N, D), np.int8)
    for n in range(N):
        for c in range(C):
            code_left = A_enc[n, c]
            dim_left = (K * c) + code_left
            out[n, dim_left] = 1
    return out

def _fit_ridge_enc(A_enc=None, Y=None, K=16, lamda=1, X_binary=None):
    """
    minimize loss of |Y - Xw|^2 + alpha * |w|^2
    X is binary in our case -> w without entry in X
    X [N, C * K]
    Y [N, C]
    W [D, C * K] -> W.T [C * K, D] later reshaped to
    [C, K, D] -> prototype dimensons
    """
    if X_binary is None:
        X_binary = sparsify_and_int8_A_enc(A_enc, K=K)
    # X_binary_sparse = csr_matrix(X_binary) # will change solver from cholesky to sparse_cg
    est = linear_model.Ridge(
        fit_intercept=False, alpha=lamda, solver="auto", copy_X=False
    )
    est.fit(X_binary, Y)
    w = est.coef_.T

    return w


def _XtA_encoded(A_enc, K=16):
    N, C = A_enc.shape
    D = C * K  # note that this is total number of centroids, not orig D

    out = np.zeros((D, D), np.int32)
    # out = np.zeros((D, D), np.float32)
    # D = int(C * K)  # note that this is total number of centroids, not orig D
    # out = np.zeros((D, D), np.int8)

    for n in range(N):
        for c in range(C):
            code_left = A_enc[n, c]
            dim_left = (K * c) + code_left
            out[dim_left, dim_left] += 1
            for cc in range(c + 1, C):
                code_right = A_enc[n, cc]
                dim_right = (K * cc) + code_right
                out[dim_left, dim_right] += 1

    # populate lower triangle
    for d in range(D):
        for dd in range(d + 1, D):
            out[dd, d] = out[d, dd]

    return out


def _XtY_encoded(A_enc, Y, K=16):
    N, C = A_enc.shape
    N, M = Y.shape

    D = int(C * K)  # note that this is total number of centroids, not orig D
    out = np.zeros((D, M), Y.dtype)

    for n in range(N):
        for c in range(C):
            code_left = A_enc[n, c]
            dim_left = (K * c) + code_left
            for m in range(M):
                out[dim_left, m] += Y[n, m]

    return out


def _XW_encoded(A_enc, W, K=16):
    N, C = A_enc.shape
    _, M = W.shape

    out = np.zeros((N, M), W.dtype)

    encoded_shifted = A_enc + np.repeat(np.arange(C) * K, N).reshape(N, C)
    for n in range(N):
        for c in range(C):
            out[n, :] += W[encoded_shifted[n, c], :]

    return out


# equation 8 in paper
def encoded_lstsq(
    A_enc=None,
    X_binary=None,
    Y=None,
    K=16,
    XtX=None,
    XtY=None,
    precondition=True,
    stable_ridge=True,
):
    if stable_ridge:
        return _fit_ridge_enc(A_enc=A_enc, Y=Y, X_binary=X_binary, K=K, lamda=1)

    if XtX is None:
        XtX = _XtA_encoded(A_enc, K=K).astype(np.float32)
        lamda = 1  # TODO cross-validate to get lamda
        # different lambda have been tested in original code
        XtX += np.diag(np.ones(XtX.shape[0]) * lamda).astype(np.float32)  # ridge

    if XtY is None:
        XtY = _XtY_encoded(A_enc, Y, K=K)

    XtX = XtX.astype(np.float64)
    XtY = XtY.astype(np.float64)

    # preconditioning to avoid numerical issues (seemingly unnecessary, but
    # might as well do it)
    if precondition:
        # different preconditions have been tested
        scale = 1.0 / np.linalg.norm(XtX, axis=0).max()
        XtX = XtX * scale
        XtY = XtY * scale

    W, _, _, _ = np.linalg.lstsq(XtX, XtY, rcond=None)  # doesn't fix it

    return W


# each codebook has const number of nonzero idxs
def _sparse_encoded_lstsq_elim_v2(
    A_enc,
    Y,
    nnz_per_centroid,
    K=16,
    # uniform_sparsity=False):  # never better
    uniform_sparsity=True,
    pq_perm_algo="start",
    stable_ridge=True,
):
    number_of_codebooks = A_enc.shape[1]
    M = Y.shape[1]
    nnz_per_centroid = min(M, int(nnz_per_centroid))
    nnz_per_centroid = max(1, nnz_per_centroid)
    assert nnz_per_centroid >= int(np.ceil(M / number_of_codebooks))
    assert nnz_per_centroid <= M

    X_binary = sparsify_and_int8_A_enc(A_enc, K=K)

    if not stable_ridge:
        # precompute XtX and XtY and create initial dense W
        XtX = _XtA_encoded(A_enc, K=K).astype(np.float32)

        lamda = 1
        # # alpha = unscaled_alpha * np.var(X - X.mean(axis=0)) * N / D
        # # lamda = np.sqrt(number_of_codebooks)
        # N = XtX.shape[0]
        # lamda = N / (K * K)
        # lamda = max(1, lamda)
        # print("using lamda = ", lamda)

        # lamda = max(1, len(A_enc) / 1e4)
        # lamda = max(1, len(A_enc) / float(K * K))
        XtX += np.diag(np.ones(XtX.shape[0]) * lamda).astype(np.float32)  # ridge
        # XtX += np.diag(np.ones(XtX.shape[0])).astype(np.float32)  # ridge
        XtY = _XtY_encoded(A_enc, Y, K=K)

        # scale = 1. / len(A_enc)
        scale = 1.0 / np.linalg.norm(XtX, axis=0).max()
        XtX = XtX * scale
        XtY = XtY * scale

        W = encoded_lstsq(
            X_binary=X_binary,
            Y=Y,
            XtX=XtX,
            XtY=XtY,
            precondition=False,
            stable_ridge=stable_ridge,
        )  # KC x M

        XtX = np.asfarray(XtX)  # since we'll be slicing columns
    else:  # stable_ridge is True
        W = encoded_lstsq(X_binary=X_binary, Y=Y, stable_ridge=stable_ridge)

    # score all blocks of W
    all_scores = np.empty((number_of_codebooks, M), dtype=np.float32)  # C x M
    for c in range(number_of_codebooks):
        Xc = A_enc[:, c].reshape(-1, 1)
        start_idx = c * K
        end_idx = start_idx + K
        Wc = W[start_idx:end_idx]

        Yc = _XtY_encoded(Xc, Wc, K=K)  # N x M
        all_scores[c] = np.linalg.norm(Yc, axis=0)

    # pq_idxs = create_codebook_start_end_idxs(M, number_of_codebooks)
    pq_idxs = create_codebook_start_end_idxs(Y, number_of_codebooks, algo=pq_perm_algo)

    # now pick which cols to keep in each codebook
    keep_mask = np.zeros((number_of_codebooks, M), dtype=np.bool_)
    # subvec_len = int(np.ceil(M / number_of_codebooks))
    for c in range(number_of_codebooks):
        # initialize with PQ
        start_idx, end_idx = pq_idxs[c]
        keep_mask[c, start_idx:end_idx] = 1

        subvec_len = end_idx - start_idx
        assert subvec_len >= 1
        keep_nidxs_extra = nnz_per_centroid - subvec_len
        scores = all_scores[c]
        scores[start_idx:end_idx] = 0

        if uniform_sparsity and keep_nidxs_extra > 0:
            # take as many other (best) nonzero idxs as we we're allowed to
            assert len(scores) >= keep_nidxs_extra
            best_idxs = np.argsort(scores)[-keep_nidxs_extra:]
            if len(best_idxs) != keep_nidxs_extra:
                print("len(best_idxs)", len(best_idxs))
                print("keep_nidxs_extra", keep_nidxs_extra)
                assert len(best_idxs) == keep_nidxs_extra
            keep_mask[c, best_idxs] = True

    if not uniform_sparsity:
        scores = all_scores.ravel()
        nkept_idxs = M  # number of nonzeros used already
        keep_nidxs_total = nnz_per_centroid * number_of_codebooks
        keep_nidxs_extra = keep_nidxs_total - nkept_idxs
        keep_idxs = np.argsort(scores)[-keep_nidxs_extra:]
        flat_mask = keep_mask.ravel()
        flat_mask[keep_idxs] = 1
        keep_mask = flat_mask.reshape(keep_mask.shape)

    # at this point, we have the mask for which cols of each centroid to keep;
    # now we just need to go from a mask to a set of indices and a sparse
    # matrix of centroids
    W_sparse = np.empty((number_of_codebooks * K, M), dtype=np.float32)
    if uniform_sparsity:
        ret_idxs = np.empty((number_of_codebooks, nnz_per_centroid), dtype=np.int64)
    else:
        ret_idxs = []
    # else:
    # ret_idxs = np.zeros((number_of_codebooks, M), dtype=np.int64) - 1
    for c in range(number_of_codebooks):
        idxs = np.where(keep_mask[c] != 0)[0]
        if uniform_sparsity:
            if len(idxs) != nnz_per_centroid:
                print("c: ", c)
                print("len(idxs): ", len(idxs))
                print("nnz_per_centroid: ", nnz_per_centroid)
                print("keep_mask counts:", keep_mask.sum(axis=1))
                assert len(idxs) == nnz_per_centroid
            ret_idxs[c] = idxs
        else:
            ret_idxs.append(idxs)

        zero_idxs = np.where(keep_mask[c] == 0)[0]
        start_idx = c * K
        end_idx = start_idx + K
        Wc = W[start_idx:end_idx]
        Wc[:, zero_idxs] = 0
        W_sparse[start_idx:end_idx] = Wc

    # now refit W_sparse to each output col; right now it's just the original
    # W with a bunch of entries zeroed
    for m in range(M):
        w = W_sparse[:, m]
        keep_idxs = np.where(w != 0)[0]

        if stable_ridge:
            X_binary_subs = X_binary[:, keep_idxs]
            w_subs = _fit_ridge_enc(X_binary=X_binary_subs, Y=Y[:, m])
        else:
            xty = XtY[:, m]
            use_XtX = XtX[keep_idxs][:, keep_idxs]
            use_xty = xty[keep_idxs]
            w_subs = np.linalg.solve(use_XtX, use_xty)
        w[:] = 0
        w[keep_idxs] = w_subs
        W_sparse[:, m] = w

    # nnzs = [len(idxs) for idxs in ret_idxs]
    # print("nnzs: ", nnzs)

    # print(f"returning {ret_idxs.shape[1]} nonzeros per centroid...")
    return W_sparse, ret_idxs


# pylint: disable=R1705
def sparse_encoded_lstsq(A_enc, Y, K=16, nnz_blocks=-1, **kwargs):
    number_of_codebooks = A_enc.shape[1]
    if nnz_blocks < 1:
        # default to returning dense centroids
        W = encoded_lstsq(A_enc, Y, K=16)
        number_of_codebooks = A_enc.shape[1]
        M = Y.shape[1]
        keep_codebook_idxs = np.empty((number_of_codebooks, M), dtype=np.int64)
        all_idxs = np.arange(M)
        for c in range(number_of_codebooks):
            keep_codebook_idxs[c] = all_idxs
        return W, keep_codebook_idxs
    else:
        nnz_per_centroid = int(nnz_blocks * Y.shape[1] / number_of_codebooks)

    return _sparse_encoded_lstsq_elim_v2(
        A_enc, Y, nnz_per_centroid=nnz_per_centroid, K=K, **kwargs
    )


def apply_hash_function(X: np.ndarray, splits: List[MultiSplit]) -> np.ndarray:
    N, _ = X.shape
    nsplits = len(splits)
    assert len(splits) >= 1
    # original code had a distinction: not sure why
    group_ids = np.zeros(N, dtype=np.int32)
    for i in range(nsplits):
        split = splits[i]
        vals = split.vals[group_ids]
        indicators = split.preprocess_x(X[:, split.dim]) > vals
        group_ids = (group_ids * 2) + indicators
    return group_ids

def maddness_encode(
    X: np.ndarray, multisplits_lists: list[list[MultiSplit]]
) -> np.ndarray:
    N, _ = X.shape
    C = len(multisplits_lists)
    A_enc = np.empty((N, C), dtype=np.int32, order="F")  # column-major
    for c in range(C):
        A_enc[:, c] = apply_hash_function(X, multisplits_lists[c])
    return np.ascontiguousarray(A_enc)

# @_memory.cache
def learn_proto_and_hash_function(
    X: np.ndarray, C: int, lut_work_const: int = -1
) -> Tuple[list[list[MultiSplit]], np.ndarray, np.ndarray]:
    _, D = X.shape
    K = 16
    used_perm_algo = "start"  # or end
    X_orig = X.astype(np.float32)

    # X_error = X_orig - centroid shape: [N, D]
    X_error, all_splits, all_prototypes, _ = init_and_learn_hash_function(
        X, C, pq_perm_algo=used_perm_algo
    )

    msv_orig = (X_orig * X_orig).mean()
    mse_error = (X_error * X_error).mean()
    print(
        "X_error mse / X mean squared value: ",
        mse_error / msv_orig,
        mse_error,
        msv_orig,
        np.mean(X_orig),
    )

    squared_diff = np.square(X_orig - X_error).mean()  # type: ignore
    print("Error to Original squared diff", squared_diff)
    # optimize prototypes discriminatively conditioned on assignments
    # applying g(A) [N, C] with values from 0-K (50000, 16)
    A_enc = maddness_encode(X, all_splits)

    # optimizing prototypes
    if lut_work_const != 1:  # if it's 1, equivalent to just doing PQ
        if lut_work_const < 0:
            # print("fitting dense lstsq to X_error")
            W = encoded_lstsq(A_enc=A_enc, Y=X_error)
        else:
            W, _ = sparse_encoded_lstsq(
                A_enc, X_error, nnz_blocks=lut_work_const, pq_perm_algo=used_perm_algo
            )

        all_prototypes_delta = W.reshape(C, K, D)
        all_prototypes += all_prototypes_delta

        # check how much improvement we got
        X_error -= _XW_encoded(A_enc, W)  # if we fit to X_error
        mse_res = (X_error * X_error).mean()

        print("X_error mse / X mse after lstsq: ", mse_res / msv_orig)

    ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        f"After Ridge regression {X.shape}-{C}-{K}"
        f"({(ram_usage / (1024 * 1024)):.3f} GB)"
    )
    
    #report_array = np.array(
    #    [
    #        mse_error,
    #        msv_orig,
    #        mse_error / msv_orig,
    #        np.mean(X_orig),
    #        mse_res,
    #        mse_res / msv_orig,
    #        ram_usage / (1024 * 1024),
    #    ]
    #)
    return all_splits, all_prototypes, None#report_array

def maddness_lut(q: np.ndarray, all_prototypes: np.ndarray) -> np.ndarray:
    q = q.reshape(1, 1, -1)  # all_prototypes is shape C, K, D
    return (q * all_prototypes).sum(axis=2)  # C, K

def maddness_quantize_luts(luts: np.ndarray, force_power_of_2: bool = True) -> Any:
    mins = luts.min(axis=(0, 2))
    maxs = luts.max(axis=(0, 2))

    gaps = maxs - mins
    gap = np.max(gaps)
    if force_power_of_2:
        exponent = np.ceil(np.log2(gap))
        scale = 2 ** int(-exponent)  # scale is a power of 2, so can just shift
        scale *= 255.5 - 1e-10  # so max val is at most 255
    else:
        scale = (255.5 - 1e-10) / gap

    offsets = mins[np.newaxis, :, np.newaxis]
    luts_quantized = (luts - offsets) * scale
    luts_quantized = (luts_quantized + 0.5).astype(np.int64)

    assert np.min(luts_quantized) >= 0
    assert np.max(luts_quantized) <= 255.0

    return luts_quantized, offsets.sum(), scale

#Ref: https://github.com/joennlae/halutmatmul/blob/master/src/python/halutmatmul/functions.py
@numba.jit(nopython=True, parallel=False)
def apply_hash_function_opt(X: np.ndarray, splits: np.ndarray) -> np.ndarray:
    N, _ = X.shape
    # original code had a distinction: not sure why
    group_ids = np.zeros(N, dtype=np.int64)  # needs to be int64 because of index :-)
    num_splits = splits.shape[0]
    length = splits.shape[1] - 3
    for i in range(num_splits):
        vals = splits[i, 0 : pow(2, i)]
        vals = vals[group_ids]
        dim = int(splits[i, length])
        scaleby = splits[i, length + 1]
        offset = splits[i, length + 2]
        x = X[:, dim] - offset
        x = x * scaleby
        indicators = x > vals
        group_ids = (group_ids * 2) + indicators
    return group_ids

#Ref: https://github.com/joennlae/halutmatmul/blob/master/src/python/halutmatmul/functions.py
@numba.jit(nopython=True, parallel=True)
def halut_encode_opt(X: np.ndarray, numpy_array: np.ndarray) -> np.ndarray:
    N, _ = X.shape
    C = numpy_array.shape[0]
    A_enc = np.empty((C, N), dtype=np.int32)  # column-major
    # split_lists = numpy_to_split_list(numpy_array)
    for c in prange(C):
        A_enc[c] = apply_hash_function_opt(X, numpy_array[c])
    return np.ascontiguousarray(A_enc.T)

#Ref: https://github.com/joennlae/halutmatmul/blob/master/src/python/halutmatmul/functions.py
def split_lists_to_numpy(
    buckets: list[list[MultiSplit]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = 0
    for c in buckets:
        for v in c:
            length = v.vals.shape[0] if v.vals.shape[0] > length else length
    i = k = 0
    ret_array = np.zeros((len(buckets), len(buckets[0]), length + 3), dtype=np.float32)
    
    thresholds = []
    dims = []
    
    # very ugly but I inherited that structure from the original code
    # which can be found here:
    # https://github.com/dblalock/bolt
    for c in buckets:
        k = 0
        for v in c:
            ret_array[i, k, 0 : pow(2, k)] = v.vals
            for t in v.vals:
                thresholds.append(t)
            ret_array[i, k, length] = v.dim
            dims.append(v.dim)
            ret_array[i, k, length + 1] = v.scaleby
            ret_array[i, k, length + 2] = v.offset
            k += 1
        i += 1
    return ret_array, np.array(thresholds, dtype=np.float32), np.array(dims)

@numba.jit(parallel=True, nopython=True)
def read_luts_opt(
    A_raveled: np.ndarray,
    A_shape: tuple[int, int],
    B_luts: np.ndarray,
    total_result: np.ndarray,
) -> np.ndarray:
    for i in prange((len(B_luts))):
        read_lut = B_luts[i].ravel()[A_raveled].reshape(A_shape)
        read_lut = read_lut.sum(axis=-1)
        total_result[i] = read_lut
    return total_result

@numba.jit(parallel=True, nopython=True)
def read_luts_quantized_opt(
    A_raveled: np.ndarray,
    A_shape: tuple[int, int],
    B_luts: np.ndarray,
    total_result: np.ndarray,
    upcast_every: int,
    C: int,
    scale: float,
    offset: float,
) -> np.ndarray:
    for i in prange((len(B_luts))):
        lut = B_luts[i]
        read_lut_1 = lut.ravel()[A_raveled].reshape(A_shape)
        shape_new = (
            read_lut_1.shape[0],
            int(A_shape[0] * A_shape[1] / upcast_every / read_lut_1.shape[0]),
            upcast_every,
        )
        read_lut = read_lut_1.reshape(shape_new)

        # while read_lut.shape[-1] > 2:
        for _ in range(np.log2(read_lut.shape[-1]) - 1):
            read_lut = (read_lut[:, :, ::2] + read_lut[:, :, 1::2] + 1) // 2

        read_lut = (read_lut[:, :, 0] + read_lut[:, :, 1] + 1) // 2
        read_lut = read_lut.sum(axis=-1)  # clipping not needed
        read_lut *= upcast_every  # convert mean to sum

        bias = C / 4 * np.log2(upcast_every)
        read_lut -= int(bias)

        read_lut = (read_lut / scale) + offset
        total_result[i] = read_lut
    return total_result

class MaddnessMatmul:
    def __init__(self, C: int = 16, lut_work_const: int = -1) -> None:
        # checks
        if lut_work_const > 0 and lut_work_const > C:
            raise Exception("lut_work_const > C: {} > {}".format(lut_work_const, C))

        self.lut_work_const = lut_work_const
        self.C = C
        self.K = 16
        self.A_enc: Optional[np.ndarray] = None
        self.luts: Optional[np.ndarray] = None
        
        self.quantize_lut = True
        self.upcast_every = 16
        self.upcast_every = min(self.C, self.upcast_every)
        # important otherwise wrong summation
        assert self.upcast_every in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        self.accumulate_how = "mean"  # sum
        
        
    def _learn_hash_buckets_and_prototypes(self, A: np.ndarray) -> None:
        _, D = A.shape
        if D < self.C:
            raise Exception("D < C: {} < {}".format(D, self.C))
        self.splits_lists, self.prototypes, _ = learn_proto_and_hash_function(
            A, self.C, lut_work_const=self.lut_work_const
        )

        self.ret_array, _, _ = split_lists_to_numpy(self.splits_lists)

    def _encode_A(self, A: np.ndarray) -> np.ndarray:
        #idxs = maddness_encode(A, self.splits_lists)
        idxs = halut_encode_opt(A, self.ret_array)
        # offsets = [  0  16  32  48  64  80  96 112 128 144 160 176 192 208 224 240]
        offsets = np.arange(self.C, dtype=np.int32) * self.K
        return idxs + offsets

    def _create_lut(self, B: np.ndarray) -> Tuple[np.ndarray, float, float]:
        B = np.atleast_2d(B)
        luts = np.zeros((B.shape[0], self.C, self.K))
        for i, q in enumerate(B):
            luts[i] = maddness_lut(q, self.prototypes)
        if self.quantize_lut:
            luts, offset, scale = maddness_quantize_luts(luts)
            return luts, offset, scale
        return luts, 0, 1

    def _calc_matmul(
        self,
        A_enc: np.ndarray,
        B_luts: np.ndarray,
        offset: float,
        scale: float,
            M=None
    ) -> np.ndarray:
        #A = np.ascontiguousarray(A_enc).ravel()
        #total_result = np.empty((len(B_luts), len(A_enc)), dtype=np.float32)
        #read_luts_quantized_opt(A, A_enc.shape, B_luts, total_result, self.upcast_every, self.C, self.scale, self.offset)

        #return total_result.T
        #upcast_every = 16
        if M is None:
            M = self.luts.shape[0]
        out =  maddness_scan(A_enc, self.C, M, self.luts).astype(np.float32)
        out = (out + offset) / scale
        # 
        #TODO: ADD BIAS
        return out #->softmax
        
    def _set_A(self, A: np.ndarray) -> None:
        self.A_enc = self._encode_A(A)

    def _set_B(self, B: np.ndarray) -> None:
        self.luts, self.offset, self.scale = self._create_lut(B.T)

    # public function
    def learn_A(self, A: np.ndarray) -> None:
        self._learn_hash_buckets_and_prototypes(A)

    def learn_offline(self, A: np.ndarray, B: np.ndarray) -> None:
        self._learn_hash_buckets_and_prototypes(A)
        self._set_B(B)

    def apply_matmul_e2e(
        self, A: np.ndarray, B: np.ndarray, A_learn: np.ndarray = None  # type: ignore
    ) -> np.ndarray:
        if A_learn is None:
            self._learn_hash_buckets_and_prototypes(A)
        else:
            self._learn_hash_buckets_and_prototypes(A_learn)
        self._set_A(A)
        self._set_B(B)
        return self._calc_matmul(
            self.A_enc,  # type: ignore[arg-type]
            self.luts,  # type: ignore[arg-type]
            offset=self.offset,
            scale=self.scale,
        )

    def matmul_online(self, A: np.ndarray) -> np.ndarray:
        self._set_A(A)
        return self._calc_matmul(
            self.A_enc, self.luts, offset=self.offset, scale=self.scale  # type: ignore[arg-type]
        )

    def reset(self) -> None:
        self.A_enc = None
        self.luts = None

    def get_speed_metrics(
        self, A: np.ndarray, B: np.ndarray, fixedA: bool = False, fixedB: bool = False
    ) -> None:
        N, D = A.shape
        D, M = B.shape
        # data encoding and LUT costs
        nmuls = 0
        nmuls += 0 if fixedA else N * D  # offset + scale before quantize
        nmuls_per_codebook_per_output = self.K * D
        nmuls_per_output = nmuls_per_codebook_per_output * self.C
        nmuls += 0 if fixedB else nmuls_per_output * M
        # lookups given encoded data + luts
        nlookups = N * M * self.C
        print("nmuls: ", nmuls, "KEY_NLOOKUPS:", nlookups)
    

if __name__ == "__main__":
    def train(
            A: np.ndarray,
            C: int = 4,
            lut_work_const: int = -1,
            A_learn: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        model = MaddnessMatmul(C=C, lut_work_const=lut_work_const)
        model._learn_hash_buckets_and_prototypes(A)
        model._set_A(A)
        return model
    N, D, M = 1280, 32, 100
    x = np.random.uniform(low=0.0, high=1.0, size=(N, D))
    model = train(x)

    y = np.random.randn(D, M)
    model._set_B(y)
    import time

    def apply_matmul(B):   
        return model._calc_matmul(
            model.A_enc,  # type: ignore[arg-type]
            model.luts,  # type: ignore[arg-type]
            offset=model.offset,
            scale=model.scale,
        )
    # Includes Tracing Time...
    result1 = apply_matmul(y)
    t1 = time.time()
    # There, remains to be optimized. (For Experiment)
    result1 = apply_matmul(y)
    t2 = time.time()
    print(t2 - t1)

    t1 = time.time()
    result1 = np.matmul(x, y)
    t2 = time.time()
    print(t2 - t1)
