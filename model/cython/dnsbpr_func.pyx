# distutils: language = c++
# cython: cdivision=True
# cython: nonecheck=False
# cython: language_level=3
import numpy as np
cimport numpy as np
from reckit.cython import is_ndarray

cdef extern from "bpr_func.h":
    void bpr_update_one_step(float* u_ptr, float* i_ptr, float* j_ptr, float* ib_ptr, float* jb_ptr,
                              int n_dim, float lr, float reg)
    float inner_product(float* a_ptr, float* b_ptr, int n_dim)


def dnsbpr_update(user_arr, pos_item_arr, neg_items_arr, lr, reg,
                  user_embeds, item_embeds, item_biases):
    if not is_ndarray(user_arr, np.int32):
        user_arr = np.array(user_arr, dtype=np.int32)
    if not is_ndarray(pos_item_arr, np.int32):
        pos_item_arr = np.array(pos_item_arr, dtype=np.int32)
    if not is_ndarray(neg_items_arr, np.int32):
        neg_items_arr = np.array(neg_items_arr, dtype=np.int32)

    if not is_ndarray(user_embeds, np.float32) \
            or not is_ndarray(item_embeds, np.float32) \
            or not is_ndarray(item_biases, np.float32):
        raise ValueError("Parameters must be np.ndarray.")

    cdef int num_pair = len(user_arr)
    cdef int num_neg = neg_items_arr.shape[-1]
    user_ptr = <int *>np.PyArray_DATA(user_arr)
    pos_item_ptr = <int *>np.PyArray_DATA(pos_item_arr)
    neg_items_ptr = <int *>np.PyArray_DATA(neg_items_arr)

    u_emb_ptr = <float *>np.PyArray_DATA(user_embeds)
    i_emb_ptr = <float *>np.PyArray_DATA(item_embeds)
    i_bias_ptr = <float *>np.PyArray_DATA(item_biases)
    cdef int num_dims = user_embeds.shape[-1]

    _dnsbpr_update(user_ptr, pos_item_ptr, neg_items_ptr, num_neg, num_pair, lr, reg,
                   u_emb_ptr, i_emb_ptr, i_bias_ptr, num_dims)


cdef void _dnsbpr_update(int* user_ptr, int* pos_item_ptr, int* neg_items_ptr, int n_neg, int num_pair, float lr, float reg,
                      float* u_emb_ptr, float* i_emb_ptr, float* i_bias_ptr, int n_dim):
    cdef int idx = 0
    cdef int user = -1
    cdef int pos_item = -1
    cdef int neg_item = -1
    cdef float* u_ptr
    cdef float* i_ptr
    cdef float* j_ptr
    cdef float* ib_ptr
    cdef float* jb_ptr

    for idx in range(num_pair):
        user = user_ptr[idx]
        pos_item = pos_item_ptr[idx]
        neg_items = neg_items_ptr + idx*n_neg

        neg_item = _item_max(user, neg_items, n_neg, u_emb_ptr, i_emb_ptr, i_bias_ptr, n_dim)
        u_ptr = u_emb_ptr+user*n_dim
        i_ptr = i_emb_ptr+pos_item*n_dim
        j_ptr = i_emb_ptr+neg_item*n_dim
        ib_ptr = i_bias_ptr+pos_item
        jb_ptr = i_bias_ptr+neg_item
        bpr_update_one_step(u_ptr, i_ptr, j_ptr, ib_ptr, jb_ptr, n_dim, lr, reg)


cdef int _item_max(int user, int* items, int n_items, float* u_emb_ptr, float* i_emb_ptr, float* i_bias_ptr, int n_dim):
        cdef int max_item = 0
        cdef float max_rating = -100000.0
        cdef int idx = 0
        cdef int item = -1
        cdef float rating = 0.0
        cdef float* u_ptr = u_emb_ptr + user*n_dim
        cdef float* i_ptr

        for idx in range(n_items):
            item = items[idx]
            i_ptr = i_emb_ptr + item*n_dim
            rating = inner_product(u_ptr, i_ptr, n_dim) + i_bias_ptr[item]
            if rating > max_rating:
                max_rating = rating
                max_item = item

        return max_item
