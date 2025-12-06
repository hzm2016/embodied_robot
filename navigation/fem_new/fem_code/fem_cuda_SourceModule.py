
from pycuda.compiler import SourceModule


def get_cuda_SourceModule():
    mod = SourceModule("""

        #include <math.h>

        typedef struct {
            int width;
            int height;
            int stride; 
            int __padding;    //为了和64位的elements指针对齐
            float* elements;
        } Matrix;
        // 设置空matrix矩阵
        __device__ Matrix SetEmptyMatrix(int height, int width, int stride) 
        {
            Matrix A;
            A.width    = width;
            A.height   = height;
            A.stride   = stride;
            A.elements = new float[width*height];
            return A;
        }
        // 读取矩阵元素
        __device__ float GetElement(const Matrix A, int row, int col)
        {
            return A.elements[row * A.stride + col];
        }

        // 赋值矩阵元素
        __device__ void SetElement(Matrix A, int row, int col, float value)
        {
            A.elements[row * A.stride + col] = value;
        }

        __device__ void matrix_multiply(Matrix *dest, Matrix *a, Matrix *b)
        {
            int i = threadIdx.x + blockDim.x * blockIdx.x;
            int j = threadIdx.y + blockDim.y * blockIdx.y;
            int width = a->width;

            float sum = 0;
            for(int k=0;k<width;k++)
            {
                sum += a->elements[i*width+k]*b->elements[k*width+j];
            }
            dest->elements[i*width+j] = sum;

        }

        __device__ void matrix_add(Matrix *dest, Matrix *a, Matrix *b)
        {
            int i = threadIdx.x + blockDim.x * blockIdx.x;
            int j = threadIdx.y + blockDim.y * blockIdx.y;

            dest->elements[i*a->width+j] = a->elements[i*a->width+j] + a->elements[i*a->width+j];
        }
        __device__ volatile int g_mutex;

        // GPU lock-based synchronization function
        __device__ void __gpu_sync(int goalVal)
        {
            // thread ID in a block
            int tid_in_block = threadIdx.x * blockDim.y + threadIdx.y;
            // only thread 0 is used for synchronization
            if (tid_in_block == 0)
            {
                atomicAdd((int*) &g_mutex, 1);

                // only when all blocks add 1 go g_mutex
                // will g_mutex equal to goalVal
                while (g_mutex != goalVal)
                {
                    // Do nothing here
                }
            }
            __syncthreads();
        }

        __device__ bool AreaMth(float *xtri, float *pNode)
        {   
            // Compute vectors

            float v[9];
            for(int m=0;m<3;m++){
                for(int n=0;n<3;n++){
                    v[m*3+n]=xtri[m*3+n]-pNode[n];
                }
            }
            //if(threadIdx.x==25&&blockIdx.x==25){printf("v is %f,%f,%f\\n", v[0], v[1], v[2]);}
            //if(threadIdx.x==25&&blockIdx.x==25){printf("v is %f,%f,%f\\n", v[3], v[4], v[5]);}
            //if(threadIdx.x==25&&blockIdx.x==25){printf("v is %f,%f,%f\\n", v[6], v[7], v[8]);}
            float vn1 = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);if(vn1==0)vn1 = 0.000000001;
            float vn2 = sqrt(v[3]*v[3]+v[4]*v[4]+v[5]*v[5]);if(vn2==0)vn2 = 0.000000001;
            float vn3 = sqrt(v[6]*v[6]+v[7]*v[7]+v[8]*v[8]);if(vn3==0)vn3 = 0.000000001;
            float theta11 = acos((v[0]*v[3]+v[1]*v[4]+v[2]*v[5])/vn1/vn2);
            float theta12 = acos((v[6]*v[3]+v[7]*v[4]+v[8]*v[5])/vn3/vn2);
            float theta13 = acos((v[0]*v[6]+v[1]*v[7]+v[2]*v[8])/vn1/vn3);

            //if(threadIdx.x==25&&blockIdx.x==250){printf("dis1 is %f,%f,%f\\n", theta11, theta12, theta13);}

            if((2 * 3.1415926535 - (theta11 + theta12 + theta13)) < 3.5){
                return 1;}
            else{
                return 0;}
        }

        __device__ float pPlane(float *Node, float *Normal, float dis0, float *pNode)
        {
            // 求投影点及点到面的距离

            float distance1 = Node[0]*Normal[0] + Node[1]*Normal[1] + Node[2]* Normal[2] - dis0;
            for(int i=0;i<3;i++){
                pNode[i] = Node[i] - distance1*Normal[i];
            }
            distance1 = fabs(distance1);
            return distance1;
        }
        /**
            dim3 bs(50, 1), grid(50, 1);
            GetAbIndex<<<grid, bs>>>(ResultNode, x, xtriangle, gNode, A1, b1, 
                                    vNormal, distance0, bool_nearpath, 
                                    constraint_triangle);
        */
        __global__ void SolveModel(Matrix *ResultNode, Matrix *x, Matrix *xtriangle, Matrix *gNode,
                                    Matrix *vNormal, Matrix *distance0,
                                    Matrix *constraint_triangle, int* Ax, Matrix *A, Matrix *b, Matrix* AdjacentTriangle, int* Adjx)
        {

            //clock_t starttime = clock();
            int i = blockIdx.x; // bool_nearpath->height||xtriangle->height
            int j = threadIdx.x; // gNode->height

            // 求解约束模型
            __shared__ float *Normal;
            __shared__ float dis0;
            if(j==blockDim.x-1){
                Normal = &vNormal->elements[i*vNormal->width];
                dis0 = distance0->elements[i];
            }
            __shared__ float xtri[9];
            if(j<9){
                xtri[j] = x->elements[int(xtriangle->elements[i*xtriangle->width+j/3])*x->width+j%3];

            }

            __syncthreads();

            float Node[3];
            for(int k=0;k<3;k++){
                Node[k]=ResultNode->elements[j*ResultNode->width+k];
            }
            float pNode[3];
            float distance1 = pPlane(Node, Normal, dis0, pNode);
            /**if(distance1< 1.5 && AreaMth(xtri, pNode)){
                int m0 = atomicAdd(Ax, 1+Adjx[i]);
                float* Normal1;
                for(int m = m0;m<m0+1+Adjx[i];m++)
                {
                    if(m==m0)
                        {Normal1 = Normal;}
                    else
                    {
                        int vn = int(AdjacentTriangle->elements[i*AdjacentTriangle->width+m-m0]);
                        Normal1 = &vNormal->elements[vn*vNormal->width];
                    }
                    int t = A->width*m;
                    int n = A->width*m + j*6;
                    for(int k = 0;k<A->width;k++){
                        A->elements[t+k]=0;
                    }
                    for(int k=0;k<3;k++){
                        A->elements[n+k] = Normal1[k];
                    }
                    b->elements[m] = dis0 - Normal1[0]*gNode->elements[j*gNode->width]-Normal1[1]*gNode->elements[j*gNode->width+1]-Normal1[2]*gNode->elements[j*gNode->width+2];
                    constraint_triangle->elements[m] = i;
                }
            }
            */
            if(distance1< 1.5 && AreaMth(xtri, pNode)){
                int m = atomicAdd(Ax, 1);
                int t = A->width*m;
                int n = A->width*m + j*6;
                for(int k = 0;k<A->width;k++){
                    A->elements[t+k]=0;
                }
                for(int k=0;k<3;k++){
                    A->elements[n+k] = - Normal[k];
                }
                b->elements[m] = -(dis0 - Normal[0]*gNode->elements[j*gNode->width]-Normal[1]*gNode->elements[j*gNode->width+1]-Normal[2]*gNode->elements[j*gNode->width+2]); //注意这里加了范围0.2
                constraint_triangle->elements[m] = i;
            }
        }


        __global__ void SearchAdjacentTriangle(Matrix *xtriangle, Matrix* AdjacentTriangle, int* Adjx)
        {
            int i = blockIdx.x; // xtriangle->height
            int j = threadIdx.x; // 1024
            __shared__ float *xtri;
            if(j==0){
                xtri = &xtriangle->elements[i*xtriangle->width];
            }

            __syncthreads();
            float kdim = gridDim.x/blockDim.x+1;
            for(int k =0;k<kdim;k++)
            {
                int m = k*blockDim.x+j;
                if(m < xtriangle->height&&m!=i)
                {
                    float* xtri2 = &xtriangle->elements[m*xtriangle->width];
                    bool a = true;
                    for(int p = 0;p<3;p++)
                    {
                        for(int q = 0;q<3;q++)
                        {
                            if(xtri[p]==xtri2[q]&&a)
                            {
                                int ax = atomicAdd(&Adjx[i], 1);
                                AdjacentTriangle->elements[i*AdjacentTriangle->width+ax] = m;
                                // 这里是错的，因为会加入本身三次，三个顶点分别和自己相等
                                a = false;
                            }
                        }
                    }
                }
            }

        }

        __global__ void TransformMatrix(Matrix *RNode, Matrix *T)
        {
            /**
            #  计算单元的坐标转换矩阵( 局部坐标 -> 整体坐标 )
            #  输入参数
            #      ie  ----- 节点号
            #  返回值
            #      T ------- 从局部坐标到整体坐标的坐标转换矩阵
            # global gElement, gNode

            xi = gNode[gElement[ie][0] - 1][0]
            yi = gNode[gElement[ie][0] - 1][1]
            zi = gNode[gElement[ie][0] - 1][2]
            xj = gNode[gElement[ie][1] - 1][0]
            yj = gNode[gElement[ie][1] - 1][1]
            zj = gNode[gElement[ie][1] - 1][2]
            L = ((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2) ** (1 / 2)
            lx = (xj - xi) / L
            mx = (yj - yi) / L
            nx = (zj - zi) / L
            */

            int i = blockIdx.x; // node_number-1
            int j = threadIdx.x; // 12
            int k = threadIdx.y; // 12

            __shared__ float L, lx, mx, nx;
            if(j==0){
                lx = (RNode->elements[(i + 1)*RNode->width] - RNode->elements[(i)*RNode->width]);
                mx = (RNode->elements[(i + 1)*RNode->width+1] - RNode->elements[(i)*RNode->width+1]);
                nx = (RNode->elements[(i + 1)*RNode->width+2] - RNode->elements[(i)*RNode->width+2]);
                L = sqrt(lx *lx + mx *mx + nx*nx);
                lx = lx/L;
                mx = mx/L;
                nx = nx/L;
            }
            __syncthreads();
            
            if(j%3==0 && j==k)T->elements[i*T->height*T->width+j*T->width+k] = lx;
            if(j%3==0 && k==j+1)T->elements[i*T->height*T->width+j*T->width+k] = -nx * lx / sqrt(lx*lx + mx*mx);
            if(j%3==0 && k==j+2)T->elements[i*T->height*T->width+j*T->width+k] = mx / sqrt(lx*lx + mx*mx);
            if(j%3==1 && k==j-1)T->elements[i*T->height*T->width+j*T->width+k] = mx;
            if(j%3==1 && j==k)T->elements[i*T->height*T->width+j*T->width+k] = -nx * mx / sqrt(lx*lx + mx*mx);
            if(j%3==1 && k==j+1)T->elements[i*T->height*T->width+j*T->width+k] = -lx / sqrt(lx*lx + mx*mx);
            if(j%3==2 && k==j-2)T->elements[i*T->height*T->width+j*T->width+k] = nx;
            if(j%3==2 && k==j-1)T->elements[i*T->height*T->width+j*T->width+k] = sqrt(lx*lx + mx*mx);
            if(j%3==2 && j==k)T->elements[i*T->height*T->width+j*T->width+k] = 0;

            /**
            T[i] = np.array(
                [[lx, -nx * lx / (lx ** 2 + mx ** 2) ** (1 / 2), mx / (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0, 0, 0,
                0, 0, 0],
                [mx, -nx * mx / (lx ** 2 + mx ** 2) ** (1 / 2), -lx / (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0, 0,
                0, 0, 0, 0],
                [nx, (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, lx, -nx * lx / (lx ** 2 + mx ** 2) ** (1 / 2), mx / (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0,
                0, 0, 0],
                [0, 0, 0, mx, -nx * mx / (lx ** 2 + mx ** 2) ** (1 / 2), -lx / (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0,
                0, 0, 0, 0],
                [0, 0, 0, nx, (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, lx, -nx * lx / (lx ** 2 + mx ** 2) ** (1 / 2), mx / (lx ** 2 + mx ** 2) ** (1 / 2),
                0, 0, 0],
                [0, 0, 0, 0, 0, 0, mx, -nx * mx / (lx ** 2 + mx ** 2) ** (1 / 2),
                -lx / (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0],
                [0, 0, 0, 0, 0, 0, nx, (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, lx, -nx * lx / (lx ** 2 + mx ** 2) ** (1 / 2),
                mx / (lx ** 2 + mx ** 2) ** (1 / 2)],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, mx, -nx * mx / (lx ** 2 + mx ** 2) ** (1 / 2),
                -lx / (lx ** 2 + mx ** 2) ** (1 / 2)],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, nx, (lx ** 2 + mx ** 2) ** (1 / 2), 0]])
            */

        }


        """)

    return mod
