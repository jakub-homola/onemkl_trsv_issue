#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>



struct MatrixCSR
{
    int32_t * rowptrs = nullptr;
    int32_t * colidxs = nullptr;
    double * vals = nullptr;
    int32_t nrows = 0;
    int32_t ncols = 0;
    int32_t nvals = 0;

    MatrixCSR() {}
    ~MatrixCSR() { clear(); }
    void resize(int32_t nrows_, int32_t ncols_, int32_t nvals_)
    {
        clear();
        nrows = nrows_;
        ncols = ncols_;
        nvals = nvals_;
        rowptrs = new int32_t[nrows + 1];
        colidxs = new int32_t[nvals];
        vals = new double[nvals];
    }
    void clear()
    {
        delete[] rowptrs;
        delete[] colidxs;
        delete[] vals;
        rowptrs = nullptr;
        colidxs = nullptr;
        vals = nullptr;
        nrows = 0;
        ncols = 0;
        nvals = 0;
    }
};

struct VectorDense
{
    double * vals = nullptr;
    int32_t size = 0;
    VectorDense() {}
    ~VectorDense() { clear(); }
    void resize(int32_t size_)
    {
        size = size_;
        vals = new double[size];
    }
    void clear()
    {
        delete[] vals;
        vals = nullptr;
        size = 0;
    }
};



bool load_matrix(MatrixCSR & output, const char * filepath)
{
    FILE * f = fopen(filepath, "r");
    if(f == nullptr)
    {
        fprintf(stderr, "Could not open file '%s'\n", filepath);
        return false;
    }

    int32_t nrows, ncols, nvals;
    fscanf(f, "%d%d%d", &nrows, &ncols, &nvals);
    output.resize(nrows, ncols, nvals);
    int32_t lastrow = 0;
    output.rowptrs[0] = 0;
    for(int32_t i = 0; i < output.nvals; i++)
    {
        int32_t row, col;
        double val;
        fscanf(f, "%d%d%lf", &row, &col, &val);
        while(row > lastrow)
        {
            lastrow++;
            output.rowptrs[lastrow] = i;
        }
        output.vals[i] = val;
        output.colidxs[i] = col;
    }
    while(nrows > lastrow)
    {
        lastrow++;
        output.rowptrs[lastrow] = nvals;
    }
    output.rowptrs[output.nrows] = output.nvals;

    fclose(f);

    return true;
}

void populate_vector(VectorDense & v, int32_t size)
{
    v.resize(size);
    for(int32_t i = 0; i < size; i++) v.vals[i] = rand() / static_cast<double>(RAND_MAX);
}
























int main(int argc, const char * * argv)
{
    if(argc <= 2)
    {
        fprintf(stderr, "Not enough arguments\n");
        return 1;
    }
    bool do_my_kernel = (atoi(argv[1]) != 0);
    int trsv_args_variant = atoi(argv[2]);

    MatrixCSR U;
    VectorDense b;

    load_matrix(U, "U.txt");
    populate_vector(b, U.nrows);



    sycl::queue q(sycl::cpu_selector{});

    {
        sycl::buffer<int32_t,1> buf_U_rowptrs(U.rowptrs, sycl::range<1>(U.nrows + 1));
        sycl::buffer<int32_t,1> buf_U_colidxs(U.colidxs, sycl::range<1>(U.nvals));
        sycl::buffer<double,1> buf_U_vals(U.vals, sycl::range<1>(U.nvals));
        sycl::buffer<double,1> buf_b(b.vals, sycl::range<1>(b.size));
        sycl::buffer<double,1> buf_b_sub(buf_b, 0, buf_b.get_range());
        sycl::buffer<double,1> buf_y(buf_b.get_range());

        oneapi::mkl::sparse::matrix_handle_t mkl_U;
        oneapi::mkl::sparse::init_matrix_handle(&mkl_U);
        oneapi::mkl::sparse::set_csr_data(mkl_U, U.nrows, U.ncols, oneapi::mkl::index_base::zero, buf_U_rowptrs, buf_U_colidxs, buf_U_vals);

        if(do_my_kernel)
        {
            printf("Submitting my modifying kernel\n");
            q.submit([&](sycl::handler & cgh) {
                auto acc_b_sub = buf_b_sub.get_access(cgh, sycl::read_write);
                cgh.single_task([=]() {
                    acc_b_sub[0] = 1;
                });
            }).wait();
        }
        else
        {
            printf("NOT submitting my modifying kernel\n");
        }

        if(trsv_args_variant == 0)
        {
            printf("Using the original arguments: subbuffer--trsv-->temporary, temporary--trsv-->subbuffer\n");
            printf("  submitting first trsv\n");
            oneapi::mkl::sparse::trsv(q, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans,    oneapi::mkl::diag::nonunit, mkl_U, buf_b_sub, buf_y);
            printf("  submitted first trsv\n");
            q.wait();
            printf("  first trsv finished\n");
            printf("  submitting second trsv\n");
            oneapi::mkl::sparse::trsv(q, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, mkl_U, buf_y, buf_b_sub);
            printf("  submitted second trsv\n");
            q.wait();
            printf("  second trsv finished\n");
        }
        else if(trsv_args_variant == 1)
        {
            printf("Using the subbuffer in both rhs and solution arguments to both calls of trsv\n");
            printf("  submitting first trsv\n");
            oneapi::mkl::sparse::trsv(q, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans,    oneapi::mkl::diag::nonunit, mkl_U, buf_b_sub, buf_b_sub);
            printf("  submitted first trsv\n");
            q.wait();
            printf("  first trsv finished\n");
            printf("  submitting second trsv\n");
            oneapi::mkl::sparse::trsv(q, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, mkl_U, buf_b_sub, buf_b_sub);
            printf("  submitted second trsv\n");
            q.wait();
            printf("  second trsv finished\n");
        }
        else if(trsv_args_variant == 2)
        {
            printf("Using regular buffer instead of a subbuffer\n");
            printf("  submitting first trsv\n");
            oneapi::mkl::sparse::trsv(q, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans,    oneapi::mkl::diag::nonunit, mkl_U, buf_b, buf_y);
            printf("  submitted first trsv\n");
            q.wait();
            printf("  first trsv finished\n");
            printf("  submitting second trsv\n");
            oneapi::mkl::sparse::trsv(q, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, mkl_U, buf_y, buf_b);
            printf("  submitted second trsv\n");
            q.wait();
            printf("  second trsv finished\n");
        }
        else if(trsv_args_variant == 3)
        {
            printf("Calling just the second trsv without the first one\n");
            printf("  submitting second trsv\n");
            oneapi::mkl::sparse::trsv(q, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, mkl_U, buf_y, buf_b_sub);
            printf("  submitted second trsv\n");
            q.wait();
            printf("  second trsv finished\n");
        }
        else
        {
            printf("Wrong variant number\n");
        }

        oneapi::mkl::sparse::release_matrix_handle(&mkl_U);
    }

    printf("End of program, everything seems OK\n");



    return 0;
}
