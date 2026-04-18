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

/*
 * lower_bound_ll: 在有序数组 arr[0..len-1] 中找 x 的第一个位置
 * 返回 1-based 编号（即离散化编号）
 * 例：b=[1,3,5,9], x=5 -> 返回 3
 */
int lower_bound_ll(long long arr[], int len, long long x) {
    int l = 0, r = len - 1, ans = len;
    while (l <= r) {
        int mid = l + ((r - l) >> 1);
        if (arr[mid] >= x) { ans = mid; r = mid - 1; }
        else                 l = mid + 1;
    }
    return ans + 1;  /* +1 转为 1-based */
}

/* ============================================================
 * 离散化
 * 将原数组 a[1..n] 的值域压缩到 [1, m]
 * 步骤：复制->排序->去重->逐元素查编号
 * ============================================================ */
void discretize() {
    /* 将 a[1..n] 复制到 b[0..n-1]（b 从 0 开始，方便 qsort） */
    for (int i = 1; i <= n; i++)
        b[i - 1] = a[i];

    qsort(b, n, sizeof(long long), cmp_ll);
    m = unique_ll(b, n);  /* m = 不同值的个数，也是值域大小 */

    /* 查每个 a[i] 对应的离散编号，存入 id[i] */
    for (int i = 1; i <= n; i++)
        id[i] = lower_bound_ll(b, m, a[i]);
}

/* ============================================================
 * update: 可持久化插入
 *
 * 含义：在旧版本 pre 的基础上，把值域位置 pos 的计数 +1，
 *       返回新版本的根节点编号。
 *
 * 参数：
 *   pre   — 旧版本根节点
 *   l, r  — 当前节点覆盖的值域区间
 *   pos   — 要插入的离散化编号（1-based）
 *
 * 实现：
 *   - 先 ++tot 分配一个新节点
 *   - 复制旧节点的 ls/rs（路径外的子树完全共享）
 *   - sum 加 1（从根到叶都要加）
 *   - 递归到 pos 所在的半边，只更新那一侧，另一侧继续共享
 * ============================================================ */
int update(int pre, int l, int r, int pos) {
    int now = ++tot;          /* 分配新节点，编号为 now */
    ls[now]  = ls[pre];       /* 默认继承旧节点的左右子指针 */
    rs[now]  = rs[pre];
    sum[now] = sum[pre] + 1;  /* 该节点覆盖的区间内多了一个元素 */

    if (l == r) return now;   /* 叶节点：直接返回，无需继续递归 */

    int mid = l + ((r - l) >> 1);
    if (pos <= mid)
        /* pos 在左半段：递归更新左子，右子继续共享旧版本 */
        ls[now] = update(ls[pre], l, mid, pos);
    else
        /* pos 在右半段：递归更新右子，左子继续共享旧版本 */
        rs[now] = update(rs[pre], mid + 1, r, pos);

    return now;
}

/* ============================================================
 * query_kth: 区间 [L, R] 第 k 小
 *
 * 利用前缀版本的差值：
 *   (root[R] 的子树) - (root[L-1] 的子树)
 *   等价于只含下标 [L, R] 内元素的权值线段树
 *
 * 在差值树上做权值线段树的经典二分：
 *   - 左子树元素个数 cnt_left = sum[ls[rightRoot]] - sum[ls[leftRoot]]
 *   - 若 k <= cnt_left，第 k 小在左半值域，递归左子
 *   - 否则第 k 小在右半值域，递归右子，k 减去 cnt_left
 *
 * 参数：
 *   leftRoot  — root[L-1]（左边界的前缀版本）
 *   rightRoot — root[R]（右边界的前缀版本）
 *   l, r      — 当前节点覆盖的值域区间
 *   k         — 求第 k 小
 *
 * 返回：离散化编号（1-based），最后用 b[pos-1] 还原真实值
 * ============================================================ */
int query_kth(int leftRoot, int rightRoot, int l, int r, int k) {
    if (l == r) return l;  /* 叶节点：当前值域只剩一个值，即为答案 */

    int mid      = l + ((r - l) >> 1);
    int cnt_left = sum[ls[rightRoot]] - sum[ls[leftRoot]]; /* 差值树左子元素数 */

    if (k <= cnt_left)
        /* 第 k 小在左半值域 */
        return query_kth(ls[leftRoot], ls[rightRoot], l, mid, k);
    else
        /* 第 k 小在右半值域，从右半找第 (k - cnt_left) 小 */
        return query_kth(rs[leftRoot], rs[rightRoot], mid + 1, r, k - cnt_left);
}

/* ============================================================
 * print_version_tree: 打印简化版版本树结构（用于验收展示）
 *
 * 输出格式示例（n=3）：
 *   ── 版本树（简化）──
 *   root[0] = 0  sum=0  (空版本)
 *   root[1] = 1  sum=1  ls=2 rs=0
 *   root[2] = 4  sum=2  ls=5 rs=0
 *   root[3] = 7  sum=3  ls=8 rs=0
 *
 * 展示内容：每个版本的根节点编号、该版本覆盖的元素总数、
 *           根节点的左右子编号（体现路径复制后的结构）
 * ============================================================ */
void print_version_tree(int n) {
    printf("\n──────── 版本树（简化展示）────────\n");
    printf("  %-8s %-10s %-8s %-8s %-8s\n",
           "版本", "根节点编号", "元素总数", "左子节点", "右子节点");
    printf("  %s\n", "─────────────────────────────────────────");

    for (int i = 0; i <= n; i++) {
        int r = root[i];
        if (i == 0) {
            printf("  root[%d] = %-5d sum=%-4d （空版本，哨兵节点）\n",
                   i, r, sum[r]);
        } else {
            printf("  root[%d] = %-5d sum=%-4d ls=%-5d rs=%-5d\n",
                   i, r, sum[r], ls[r], rs[r]);
        }
    }
    printf("  （共分配节点数 tot = %d,每版本新增约 log2(%d)≈%d 个节点）\n\n",
           tot, m, /* 粗略估算 log */ 
           /* 手动计算 log2 避免引入 math.h */
           m > 0 ? (m > 1 ? (m > 3 ? (m > 7 ? (m > 15 ? 5 : 4) : 3) : 2) : 1) : 0);
}

/* ============================================================
 * 主函数
 * ============================================================ */
int main() {
    int q;

    /* 读入数组 */
    scanf("%d", &n);
    for (int i = 1; i <= n; i++)
        scanf("%lld", &a[i]);

    /* 第一步：离散化 */
    discretize();

    /* 第二步：建前缀版本树
     * root[0] = 0（哨兵空版本）
     * root[i] = 在 root[i-1] 基础上插入 id[i]（a[i] 的离散编号）
     */
    root[0] = 0;
    for (int i = 1; i <= n; i++)
        root[i] = update(root[i - 1], 1, m, id[i]);

    /* 打印版本树结构（便于验收展示） */
    print_version_tree(n);

    /* 第三步：回答查询 */
    scanf("%d", &q);
    printf("──────── 查询结果 ────────\n");

    while (q--) {
        int l, r, k;
        scanf("%d %d %d", &l, &r, &k);

        /* 合法性检查：k 不能超过区间内元素个数 */
        int cnt = sum[root[r]] - sum[root[l - 1]];
        if (k < 1 || k > cnt) {
            printf("查询 [%d,%d] 第%d小：k 越界（区间共%d个元素）\n",
                   l, r, k, cnt);
            continue;
        }

        /* 查询离散编号，再还原真实值 */
        int pos = query_kth(root[l - 1], root[r], 1, m, k);
        printf("区间[%d,%d]的第%d小 = %lld\n", l, r, k, b[pos - 1]);
    }

    return 0;
}