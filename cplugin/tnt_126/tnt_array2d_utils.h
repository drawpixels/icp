/*
*
* Template Numerical Toolkit (TNT)
*
* Mathematical and Computational Sciences Division
* National Institute of Technology,
* Gaithersburg, MD USA
*
*
* This software was developed at the National Institute of Standards and
* Technology (NIST) by employees of the Federal Government in the course
* of their official duties. Pursuant to title 17 Section 105 of the
* United States Code, this software is not subject to copyright protection
* and is in the public domain. NIST assumes no responsibility whatsoever for
* its use by other parties, and makes no guarantees, expressed or implied,
* about its quality, reliability, or any other characteristic.
*
*/


#ifndef TNT_ARRAY2D_UTILS_H
#define TNT_ARRAY2D_UTILS_H

#include <cstdlib>
#include <cassert>
#include <tnt_array1d.h>

namespace TNT
{


template <class T>
std::ostream& operator<<(std::ostream &s, const Array2D<T> &A)
{
    int M=A.dim1();
    int N=A.dim2();

    s << M << " " << N << "\n";

    for (int i=0; i<M; i++)
    {
        for (int j=0; j<N; j++)
        {
            s << A[i][j] << " ";
        }
        s << "\n";
    }


    return s;
}

template <class T>
std::istream& operator>>(std::istream &s, Array2D<T> &A)
{

    int M, N;

    s >> M >> N;

	Array2D<T> B(M,N);

    for (int i=0; i<M; i++)
        for (int j=0; j<N; j++)
        {
            s >>  B[i][j];
        }

	A = B;
    return s;
}


template <class T>
Array2D<T> operator+(const Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() != m ||  B.dim2() != n )
		return Array2D<T>();

	else
	{
		Array2D<T> C(m,n);

		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				C[i][j] = A[i][j] + B[i][j];
		}
		return C;
	}
}

template <class T>
Array2D<T> operator-(const Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() != m ||  B.dim2() != n )
		return Array2D<T>();

	else
	{
		Array2D<T> C(m,n);

		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				C[i][j] = A[i][j] - B[i][j];
		}
		return C;
	}
}


template <class T>
Array2D<T> operator*(const Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() != m ||  B.dim2() != n )
		return Array2D<T>();

	else
	{
		Array2D<T> C(m,n);

		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				C[i][j] = A[i][j] * B[i][j];
		}
		return C;
	}
}




template <class T>
Array2D<T> operator/(const Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() != m ||  B.dim2() != n )
		return Array2D<T>();

	else
	{
		Array2D<T> C(m,n);

		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				C[i][j] = A[i][j] / B[i][j];
		}
		return C;
	}
}

//******** NEW FUNCTIONS (begin)
    
    template <class T>
    Array2D<T> operator+(const Array2D<T> &A, const T B)
    {
        int m = A.dim1();
        int n = A.dim2();
        
        Array2D<T> C(m,n);
        
        for (int i=0; i<m; i++)
        {
            for (int j=0; j<n; j++)
                C[i][j] = A[i][j] + B;
        }
        return C;
    }
    
    template <class T>
    Array2D<T> operator-(const Array2D<T> &A, const T B)
    {
        int m = A.dim1();
        int n = A.dim2();
        
        Array2D<T> C(m,n);
        
        for (int i=0; i<m; i++)
        {
            for (int j=0; j<n; j++)
                C[i][j] = A[i][j] - B;
        }
        return C;
    }
    
    template <class T>
    Array2D<T> operator*(const Array2D<T> &A, const T B)
    {
        int m = A.dim1();
        int n = A.dim2();
        
        Array2D<T> C(m,n);
        
        for (int i=0; i<m; i++)
        {
            for (int j=0; j<n; j++)
                C[i][j] = A[i][j] * B;
        }
        return C;
    }
    
    template <class T>
    Array2D<T> operator/(const Array2D<T> &A, const T B)
    {
        int m = A.dim1();
        int n = A.dim2();
        
        if (B == 0 )
            return Array2D<T>();
        
        else
        {
            Array2D<T> C(m,n);
            
            for (int i=0; i<m; i++)
            {
                for (int j=0; j<n; j++)
                    C[i][j] = A[i][j] / B;
            }
            return C;
        }
    }
    
//******** NEW FUNCTIONS (end)



template <class T>
Array2D<T>&  operator+=(Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() == m ||  B.dim2() == n )
	{
		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				A[i][j] += B[i][j];
		}
	}
	return A;
}



template <class T>
Array2D<T>&  operator-=(Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() == m ||  B.dim2() == n )
	{
		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				A[i][j] -= B[i][j];
		}
	}
	return A;
}



template <class T>
Array2D<T>&  operator*=(Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() == m ||  B.dim2() == n )
	{
		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				A[i][j] *= B[i][j];
		}
	}
	return A;
}





template <class T>
Array2D<T>&  operator/=(Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() == m ||  B.dim2() == n )
	{
		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				A[i][j] /= B[i][j];
		}
	}
	return A;
}


//********** NEW FUNCTIONS (begin)

    template <class T>
    Array2D<T>&  operator+=(Array2D<T> &A, const T B)
    {
        int m = A.dim1();
        int n = A.dim2();
        
        for (int i=0; i<m; i++)
        {
            for (int j=0; j<n; j++)
                A[i][j] += B;
        }
        return A;
    }
    
    template <class T>
    Array2D<T>&  operator-=(Array2D<T> &A, const T B)
    {
        int m = A.dim1();
        int n = A.dim2();
        
        for (int i=0; i<m; i++)
        {
            for (int j=0; j<n; j++)
                A[i][j] -= B;
        }
        return A;
    }

    template <class T>
    Array2D<T>&  operator*=(Array2D<T> &A, const T B)
    {
        int m = A.dim1();
        int n = A.dim2();
        
        for (int i=0; i<m; i++)
        {
            for (int j=0; j<n; j++)
                A[i][j] *= B;
        }
        return A;
    }
    
    template <class T>
    Array2D<T>&  operator/=(Array2D<T> &A, const T B)
    {
        int m = A.dim1();
        int n = A.dim2();
        
        if (B != 0 )
        {
            for (int i=0; i<m; i++)
            {
                for (int j=0; j<n; j++)
                    A[i][j] /= B;
            }
        }
        return A;
    }
    
//********** NEW FUNCTIONS (end)
    
    
    /**
     Matrix Multiply:  compute C = A*B, where C[i][j]
     is the dot-product of row i of A and column j of B.
     
     
     @param A an (m x n) array
     @param B an (n x k) array
     @return the (m x k) array A*B, or a null array (0x0)
     if the matrices are non-conformant (i.e. the number
     of columns of A are different than the number of rows of B.)
     
     
     */
    template <class T>
    Array2D<T> matmult(const Array2D<T> &A, const Array2D<T> &B)
    {
        if (A.dim2() != B.dim1())
            return Array2D<T>();
        
        int M = A.dim1();
        int N = A.dim2();
        int K = B.dim2();
        
        Array2D<T> C(M,K);
        
        for (int i=0; i<M; i++)
            for (int j=0; j<K; j++)
            {
                T sum = 0;
                
                for (int k=0; k<N; k++)
                    sum += A[i][k] * B [k][j];
                
                C[i][j] = sum;
            }
        
        return C;
        
    }

//****** NEW FUNCTION (begin)

    // Extract one row of the matrix as Array1D
    template <class T>
    Array1D<T> matrow (const Array2D<T> &A, const int row)
    {
        // Invalid row
        if ((row<0) || (row>=A.dim1()))
            return Array1D<T>();
        
        return Array1D<T> (A.dim2(), A[row]);
    }

    // Extract one column of the matrix as Array1D
    template <class T>
    Array1D<T> matcolumn (const Array2D<T> &A, const int column)
    {
        // Invalid column
        if ((column<0) || (column>=A.dim2()))
            return Array1D<T>();
        
        Array1D<T> C(A.dim2());
        for (int i=0; i<A.dim1(); i++)
            C[i] = A[i][column];
        
        return C;
    }
    
    // Pre-multiplication between Matrix and COLUMN Vector
    template <class T>
    Array1D<T> matmult(const Array2D<T> &A, const Array1D<T> &B)
    {
        if (A.dim2() != B.dim1())
            return Array1D<T>();
        
        int M = A.dim1();
        int N = A.dim2();
        
        Array1D<T> C(M);
        
        for (int i=0; i<M; i++) {
            T sum = 0;
            
            for (int j=0; j<N; j++) {
                sum += A[i][j] * B [j];
            }

            C[i] = sum;
        }
        
        return C;
        
    }

    // Post-multiplication between ROW Vector and Matrix
    template <class T>
    Array1D<T> matmult(const Array1D<T> &A, const Array2D<T> &B)
    {
        if (A.dim1() != B.dim1())
            return Array1D<T>();
        
        int M = B.dim1();
        int N = B.dim2();
        
        Array1D<T> C(N);
        
        for (int i=0; i<N; i++) {
            T sum = 0;
            
            for (int j=0; j<M; j++) {
                sum += A[j] * B[j][i];
            }
            
            C[i] = sum;
        }
        
        return C;
        
    }
    
    
    // Dot product
    template <class T>
    T dot(const Array1D<T> &A, const Array1D<T> &B)
    {
        if (A.dim1() != B.dim1())
            return 0;
        
        int M = A.dim1();
        
        T sum = 0;
        for (int i=0; i<M; i++)
            sum += A[i] * B [i];
        
        return sum;
        
    }
    
    template <class T>
    Array2D<T> addcolumn(const Array2D<T> &A, const Array1D<T> &B)
    {
        int m = A.dim1();
        int n = A.dim2();
        
        if (B.dim1() != m)
            return Array2D<T>();
        
        else
        {
            Array2D<T> C(m,n);
            
            for (int i=0; i<m; i++)
            {
                for (int j=0; j<n; j++)
                    C[i][j] = A[i][j] + B[i];
            }
            return C;
        }
    }

    template <class T>
    Array2D<T> addrow(const Array2D<T> &A, const Array1D<T> &B)
    {
        int m = A.dim1();
        int n = A.dim2();
        
        if (B.dim1() != n)
            return Array2D<T>();
        
        else
        {
            Array2D<T> C(m,n);
            
            for (int i=0; i<m; i++)
            {
                for (int j=0; j<n; j++)
                    C[i][j] = A[i][j] + B[j];
            }
            return C;
        }
    }
    
//****** NEW FUNCTION (end)


} // namespace TNT

#endif
