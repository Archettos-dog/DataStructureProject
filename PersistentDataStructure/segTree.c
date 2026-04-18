#include <stdio.h>
#include <stdlib.h>

/* ============================================================
 * 可持久化权值线段树（主席树）—— 区间第 k 小查询
 *
 * 核心思路：
 *   1. 离散化：将原数组值域压缩到 [1, m]，节省空间
 *   2. 权值线段树：维护"值域上每个值出现了几次"
 *   3. 可持久化：对 a[1..n] 逐个插入，每次只复制被修改的
 *      路径（O(log m) 个新节点），其余节点与旧版本共享
 *   4. 区间查询：root[R] - root[L-1] 的差值树等价于
 *      只统计下标 [L,R] 内元素的权值线段树，再在其上二分
 *
 * 时间复杂度：建树 O(n log m)，单次查询 O(log m)
 * 空间复杂度：O(n log m)（每个版本新增 O(log m) 节点）
 * ============================================================ */

#define MAXN    200005          /* 数组最大长度 */
#define MAXNODE (MAXN * 40)     /* 节点池大小：每次插入最多新建 log(MAXN)≈18 个节点
                                   留 40 倍余量足够 */

/* ---------- 原数组与离散化 ---------- */
long long a[MAXN];   /* 原始输入数组（1-indexed） */
long long b[MAXN];   /* 离散化辅助数组：存排序后的去重值 */
int       id[MAXN];  /* id[i] = a[i] 在离散化数组 b 中的 1-based 编号 */

/* ---------- 主席树节点池 ---------- */
/*
 * 所有版本的节点共用一个静态池，用 tot 计数已分配节点数。
 * 节点 0 是哨兵节点（ls=rs=sum=0），代表空树，永远不修改它。
 */
int root[MAXN];      /* root[i] = 前 i 个元素构成的版本的根节点编号 */
int ls[MAXNODE];     /* ls[u] = 节点 u 的左子节点编号 */
int rs[MAXNODE];     /* rs[u] = 节点 u 的右子节点编号 */
int sum[MAXNODE];    /* sum[u] = 节点 u 覆盖的值域区间内，已插入元素的总数 */
int tot = 0;         /* 已分配节点总数（0 号节点为哨兵，从 1 开始分配） */

int n, m;            /* n=数组长度，m=离散化后不同值的个数（值域大小） */

/* ============================================================
 * 工具函数：比较、去重、二分
 * ============================================================ */

/* qsort 的比较函数：按 long long 升序 */
int cmp_ll(const void *x, const void *y) {
    long long va = *(const long long *)x;
    long long vb = *(const long long *)y;
    if (va < vb) return -1;
    if (va > vb) return  1;
    return 0;
}

/*
 * unique_ll: 对已排序数组原地去重，返回去重后长度
 * 例：[1,1,3,5,5] -> [1,3,5]，返回 3
 */
int unique_ll(long long arr[], int len) {
    int k = 0;
    for (int i = 0; i < len; i++) {
        if (i == 0 || arr[i] != arr[i - 1])
            arr[k++] = arr[i];
    }
    return k;
}