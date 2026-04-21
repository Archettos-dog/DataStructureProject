/* ============================================================
 * 可持久化线段树优化
 *
 * 支持的查询类型（输入格式见主函数）：
 *   类型1：区间 [L,R] 第 k 小
 *   类型2：区间 [L,R] 不同元素个数
 *   类型3：区间 [L,R] 众数（出现次数最多的值，多个取最小）
 *   类型4：值 v 在区间 [L,R] 中出现的次数
 *
 * 优化点：
 *   1. 两棵树合并为一个节点池，节省约一半内存
 *   2. MAXNODE 基于精确的空间复杂度分析推导
 *   3. 完整的输入合法性检查与错误提示
 *
 * 时间复杂度：建树 O(n log n)，每次查询 O(log n)
 *
 * 空间复杂度分析（决定 MAXNODE 大小）：
 *   设 n = 数组长度，m = 离散化后不同值的个数（m <= n）
 *
 *   树1（权值线段树，值域 [1,m]）：
 *     每次 update 新建恰好 floor(log2(m))+1 个节点
 *     共 n 次插入，节点上界 = n * (floor(log2(m))+1)
 *     n=200000, log2(200000)<18 => 上界 = 200000*18 = 3,600,000
 *
 *   树2（位置线段树，位置 [1,n]）：
 *     每次 update 最多两次调用（删旧位置 + 加新位置）
 *     每次 update 新建 floor(log2(n))+1 个节点
 *     节点上界 = 2 * n * (floor(log2(n))+1)
 *     n=200000, log2(200000)<18 => 上界 = 2*200000*18 = 7,200,000
 *
 *   两棵树共用一个节点池，总上界 = 3,600,000 + 7,200,000 = 10,800,000
 *   取 MAXN*60 = 12,000,000，保留约 10% 的余量
 * ============================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- 容量常量 ---------- */
#define MAXN    200005

/*
 * 合并节点池大小：
 *   树1上界 n*18=3,600,000 + 树2上界 2*n*18=7,200,000
 *   = 10,800,000，取 MAXN*60=12,000,000 留余量
 */
#define MAXNODE (MAXN * 60)

/* ---------- 原数组与离散化 ---------- */
long long a[MAXN];   /* 原始数组（1-indexed） */
long long b[MAXN];   /* 排序去重后的离散化值表 */
int       id[MAXN];  /* id[i] = a[i] 在 b 中的 1-based 编号 */
int       n, m;      /* n=数组长度，m=不同值个数（值域大小） */

/* ============================================================
 * 合并节点池
 * 树1（root1）和树2（root2）共用 ls/rs/sum 三个数组
 * 两棵树的节点编号在同一空间内互不冲突（tot 统一递增分配）
 * ============================================================ */
int ls[MAXNODE];     /* ls[u] = 节点 u 的左子节点编号 */
int rs[MAXNODE];     /* rs[u] = 节点 u 的右子节点编号 */
int sum[MAXNODE];    /* sum[u] = 节点 u 覆盖区间内的计数 */
int tot = 0;         /* 已分配节点数（0 为哨兵节点，从 1 开始分配） */

int root1[MAXN];     /* root1[i] = 树1第 i 个版本的根（前 i 个元素） */
int root2[MAXN];     /* root2[i] = 树2第 i 个版本的根（处理完前 i 个元素） */

/* prev_occ[v] = 离散值 v 上一次出现的位置（0 表示从未出现） */
int prev_occ[MAXN];

/* ============================================================
 * 工具函数
 * ============================================================ */

/* qsort 的 long long 升序比较函数 */
int cmp_ll(const void *x, const void *y) {
    long long va = *(const long long *)x;
    long long vb = *(const long long *)y;
    return (va > vb) - (va < vb);
}

/* 对已排序数组原地去重，返回去重后长度 */
int unique_ll(long long arr[], int len) {
    int k = 0;
    for (int i = 0; i < len; i++)
        if (i == 0 || arr[i] != arr[i - 1])
            arr[k++] = arr[i];
    return k;
}

/*
 * lower_bound_ll: 在有序数组 arr[0..len-1] 中找 x 的第一个位置
 * 返回 1-based 编号（即离散化编号）
 */
int lower_bound_ll(long long arr[], int len, long long x) {
    int l = 0, r = len - 1, ans = len;
    while (l <= r) {
        int mid = l + ((r - l) >> 1);
        if (arr[mid] >= x) { ans = mid; r = mid - 1; }
        else                  l = mid + 1;
    }
    return ans + 1;
}

/* 离散化：将 a[1..n] 压缩到 [1,m] */
void discretize() {
    for (int i = 1; i <= n; i++) b[i - 1] = a[i];
    qsort(b, n, sizeof(long long), cmp_ll);
    m = unique_ll(b, n);
    for (int i = 1; i <= n; i++)
        id[i] = lower_bound_ll(b, m, a[i]);
}

/* ============================================================
 * 通用 update：在旧版本 pre 基础上，位置 pos 加 delta
 *
 * 树1调用：delta = +1，坐标系为值域 [1,m]
 * 树2调用：delta = +1（新增标记）或 -1（删除旧标记），坐标系为位置 [1,n]
 *
 * 路径复制：只复制从根到 pos 的路径节点（O(log n) 个），
 *           路径外的子树直接共享旧版本的节点指针
 * ============================================================ */
int update(int pre, int l, int r, int pos, int delta) {
    /* 节点池溢出检测：确保不越界 */
    if (tot >= MAXNODE - 1) {
        fprintf(stderr, "Fatal: node pool overflow (tot=%d)\n", tot);
        exit(1);
    }
    int now = ++tot;
    ls[now]  = ls[pre];
    rs[now]  = rs[pre];
    sum[now] = sum[pre] + delta;
    if (l == r) return now;
    int mid = l + ((r - l) >> 1);
    if (pos <= mid)
        ls[now] = update(ls[pre], l, mid, pos, delta);
    else
        rs[now] = update(rs[pre], mid + 1, r, pos, delta);
    return now;
}

/* ============================================================
 * 查询函数 1：区间 [L,R] 第 k 小
 *
 * 利用前缀差值：root1[R] - root1[L-1] 等价于只含 [L,R]
 * 内元素的权值分布，在差值树上做值域二分
 *
 * 参数：
 *   lrt = root1[L-1]，rrt = root1[R]
 *   l,r = 当前节点覆盖的值域区间
 *   k   = 求第 k 小
 * 返回：离散化编号，主函数用 b[pos-1] 还原真实值
 * ============================================================ */
int query_kth(int lrt, int rrt, int l, int r, int k) {
    if (l == r) return l;
    int mid      = l + ((r - l) >> 1);
    int cnt_left = sum[ls[rrt]] - sum[ls[lrt]];
    if (k <= cnt_left)
        return query_kth(ls[lrt], ls[rrt], l, mid, k);
    else
        return query_kth(rs[lrt], rs[rrt], mid + 1, r, k - cnt_left);
}

/* ============================================================
 * 查询函数 2：区间 [L,R] 不同元素个数
 *
 * 只需传入 root2[R]（单棵树），不做差值。
 * root2[R] 保证每种值只在其最后出现的位置上有标记，
 * 统计 [L,R] 内的活跃标记数即为不同元素个数。
 *
 * 参数：
 *   node    = root2[R]
 *   l,r     = 当前节点覆盖的位置区间
 *   ql,qr   = 查询的位置区间 [L,R]
 * ============================================================ */
int query_distinct(int node, int l, int r, int ql, int qr) {
    if (ql <= l && r <= qr) return sum[node];
    int mid = l + ((r - l) >> 1);
    int res = 0;
    if (ql <= mid) res += query_distinct(ls[node], l, mid, ql, qr);
    if (qr >  mid) res += query_distinct(rs[node], mid + 1, r, ql, qr);
    return res;
}

/* ============================================================
 * 查询函数 3：区间 [L,R] 众数
 *
 * 在差值树上，贪心地走向计数更多的一侧。
 * 若左右计数相等，走左侧（保证返回最小的众数）。
 *
 * 参数：
 *   lrt = root1[L-1]，rrt = root1[R]
 *   l,r = 当前节点覆盖的值域区间
 * 返回：离散化编号，主函数用 b[pos-1] 还原真实值
 * ============================================================ */
int query_mode(int lrt, int rrt, int l, int r) {
    if (l == r) return l;
    int mid       = l + ((r - l) >> 1);
    int cnt_left  = sum[ls[rrt]] - sum[ls[lrt]];
    int cnt_right = sum[rs[rrt]] - sum[rs[lrt]];
    /*
     * 走计数更大的一侧；相等时走左侧，
     * 因为左侧覆盖更小的值域，保证取到最小的众数
     */
    if (cnt_left >= cnt_right)
        return query_mode(ls[lrt], ls[rrt], l, mid);
    else
        return query_mode(rs[lrt], rs[rrt], mid + 1, r);
}

/* ============================================================
 * 查询函数 4：值 v 在区间 [L,R] 中出现的次数
 *
 * 在差值树上找值域位置 pos（v 的离散编号）的叶节点，
 * 读取其 sum 差值即为出现次数。
 *
 * 参数：
 *   lrt = root1[L-1]，rrt = root1[R]
 *   l,r = 当前节点覆盖的值域区间
 *   pos = v 的离散化编号
 * ============================================================ */
int query_count(int lrt, int rrt, int l, int r, int pos) {
    if (l == r) return sum[rrt] - sum[lrt];
    int mid = l + ((r - l) >> 1);
    if (pos <= mid)
        return query_count(ls[lrt], ls[rrt], l, mid, pos);
    else
        return query_count(rs[lrt], rs[rrt], mid + 1, r, pos);
}

/* ============================================================
 * 版本树展示
 * ============================================================ */
void print_version_tree() {
    printf("\n-------- Version Tree --------\n");
    printf("  %-5s  %-12s %-8s  %-12s %-8s\n",
           "Ver", "Tree1 root", "T1 cnt", "Tree2 root", "T2 active");
    printf("  %-5s  %-12s %-8s  %-12s %-8s\n",
           "-----","----------","------","----------","--------");
    for (int i = 0; i <= n; i++) {
        int r1 = root1[i], r2 = root2[i];
        printf("  [%2d]  %-12d %-8d  %-12d %-8d",
               i, r1, sum[r1], r2, sum[r2]);
        if (i == 0) printf("  <- empty");
        else        printf("  <- insert a[%d]=%lld", i, a[i]);
        printf("\n");
    }
    printf("  (shared pool: tot=%d / %d)\n\n", tot, MAXNODE);
}

/* ============================================================
 * 主函数
 *
 * 输入格式：
 *   第1行：n（数组长度）
 *   第2行：n 个整数（数组元素）
 *   第3行：q（查询次数）
 *   之后 q 行，每行以类型编号开头：
 *     1 L R K   —— 区间[L,R]第K小
 *     2 L R     —— 区间[L,R]不同元素个数
 *     3 L R     —— 区间[L,R]众数（多个众数取最小）
 *     4 L R V   —— 值V在区间[L,R]中出现的次数
 * ============================================================ */
int main() {
    /* 读入并校验 n */
    if (scanf("%d", &n) != 1 || n <= 0 || n > 200000) {
        fprintf(stderr, "Error: invalid n\n");
        return 1;
    }
    for (int i = 1; i <= n; i++) {
        if (scanf("%lld", &a[i]) != 1) {
            fprintf(stderr, "Error: failed to read a[%d]\n", i);
            return 1;
        }
    }

    /* 第一步：离散化 */
    discretize();

    /* 第二步：同时建两棵可持久化线段树 */
    root1[0] = 0;
    root2[0] = 0;
    memset(prev_occ, 0, sizeof(prev_occ));

    for (int i = 1; i <= n; i++) {
        /* 树1：值域位置 id[i] 计数 +1 */
        root1[i] = update(root1[i - 1], 1, m, id[i], +1);

        /*
         * 树2：维护"每种值只在最后一次出现位置上有活跃标记"
         *   若 id[i] 之前出现过：旧位置 -1，当前位置 +1
         *   若首次出现：当前位置 +1
         */
        if (prev_occ[id[i]] != 0) {
            int tmp  = update(root2[i - 1], 1, n, prev_occ[id[i]], -1);
            root2[i] = update(tmp,          1, n, i,                +1);
        } else {
            root2[i] = update(root2[i - 1], 1, n, i, +1);
        }
        prev_occ[id[i]] = i;
    }

    /* 展示版本树 */
    print_version_tree();

    /* 第三步：回答查询 */
    int q;
    if (scanf("%d", &q) != 1 || q <= 0) {
        fprintf(stderr, "Error: invalid q\n");
        return 1;
    }

    printf("-------- Query Results --------\n");

    while (q--) {
        int type, l, r;
        if (scanf("%d %d %d", &type, &l, &r) != 3) {
            fprintf(stderr, "Error: failed to read query\n");
            return 1;
        }

        /* 通用区间合法性检查 */
        if (l < 1 || r > n || l > r) {
            printf("Error: invalid range [%d,%d] for n=%d\n", l, r, n);
            continue;
        }

        if (type == 1) {
            /* ---- 功能1：区间第 k 小 ---- */
            int k;
            if (scanf("%d", &k) != 1) {
                fprintf(stderr, "Error: failed to read k\n");
                return 1;
            }
            int cnt = sum[root1[r]] - sum[root1[l - 1]];
            if (k < 1 || k > cnt) {
                printf("Query1 [%d,%d] k=%d: k out of range (size=%d)\n",
                       l, r, k, cnt);
                continue;
            }
            int pos = query_kth(root1[l - 1], root1[r], 1, m, k);
            printf("Query1: [%d,%d] kth(%d) = %lld\n", l, r, k, b[pos - 1]);

        } else if (type == 2) {
            /* ---- 功能2：区间不同元素个数 ---- */
            int ans = query_distinct(root2[r], 1, n, l, r);
            printf("Query2: [%d,%d] distinct count = %d\n", l, r, ans);

        } else if (type == 3) {
            /* ---- 功能3：区间众数 ---- */
            int pos     = query_mode(root1[l - 1], root1[r], 1, m);
            int mode_cnt = sum[root1[r]] - sum[root1[l - 1]]; /* 区间总元素数，仅供参考 */
            /*
             * 还原众数出现次数：在差值树上查叶节点 pos 的 sum 差值
             * 直接复用 query_count
             */
            int freq = query_count(root1[l - 1], root1[r], 1, m, pos);
            printf("Query3: [%d,%d] mode = %lld (appears %d times, range size=%d)\n",
                   l, r, b[pos - 1], freq, mode_cnt);

        } else if (type == 4) {
            /* ---- 功能4：值 v 在区间内的出现次数 ---- */
            long long v;
            if (scanf("%lld", &v) != 1) {
                fprintf(stderr, "Error: failed to read v\n");
                return 1;
            }
            /* 检查 v 是否在离散化表中（即是否在原数组中出现过） */
            int pos = lower_bound_ll(b, m, v);
            if (pos < 1 || pos > m || b[pos - 1] != v) {
                printf("Query4: [%d,%d] count(%lld) = 0 (value not in array)\n",
                       l, r, v);
                continue;
            }
            int cnt = query_count(root1[l - 1], root1[r], 1, m, pos);
            printf("Query4: [%d,%d] count(%lld) = %d\n", l, r, v, cnt);

        } else {
            printf("Error: unknown query type %d\n", type);
        }
    }

    return 0;
}