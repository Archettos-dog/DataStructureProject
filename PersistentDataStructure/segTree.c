#define MAXN    200005         
#define MAXNODE (MAXN * 40)     

// 全局数组和变量
int a[MAXN];        // 原数组
int b[MAXN];        // 离散化用的辅助数组
int root[MAXN];     // 每个前缀对应一棵树的根
int ls[MAXNODE];    // 左孩子
int rs[MAXNODE];    // 右孩子
int sum[MAXNODE];   // 当前区间内数字出现次数
int tot;            // 当前已经用了多少个节点
int n, m;           // n 是数组长度，m 是去重后值域大小

// 1. 离散化
void discretize();

// 2. 插入一个数，生成新版本
int update(int pre, int l, int r, int x);

// 3. 查询区间 [l, r] 中第 k 小
int query_kth(int leftRoot, int rightRoot, int l, int r, int k);

// 4. 主函数
int main();