! Copyright (c) 2018, Susi Lehtola
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!
! 1. Redistributions of source code must retain the above copyright notice, this
!    list of conditions and the following disclaimer.
! 2. Redistributions in binary form must reproduce the above copyright notice,
!    this list of conditions and the following disclaimer in the documentation
!    and/or other materials provided with the distribution.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
! ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
! WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
! ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
! (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
! LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
! ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
! (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
! SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

module legendre_wrapper_m
  use Associated_Legendre_Functions
  use iso_c_binding
  implicit none
contains
  ! Main worker routines: calculate whole array of Plm and Qlm
  subroutine calculate_Plm_array(Plm,xi)
    double precision, dimension(:,:), intent(out) :: Plm
    double precision, intent(in) :: xi

    allocate(x(1))
    x(1)=xi

    directive   = 'regular'
    Leg%D%A%Dir = 'Miller'
    l_max = size(Plm,1)-1
    m_min = 0
    m_max = size(Plm,2)-1
    if(m_max > l_max) l_max=m_max

    allocate(Leg%R_LM%F(0:l_max,0:m_max))
    call Legendre( R_LM=Leg%R_LM )
    Plm(1:size(Plm,1),:) = Leg%R_LM%F(0:size(Plm,1)-1,:)
    deallocate(Leg%R_LM%F)
    deallocate(x)
  end subroutine calculate_Plm_array

  subroutine calculate_Qlm_array(Qlm,xi)
    double precision, dimension(:,:), intent(out) :: Qlm
    double precision, intent(in) :: xi

    allocate(x(1))
    x(1)=xi

    directive   = 'irregular'
    Leg%D%A%Dir = 'Miller'
    l_max = size(Qlm,1)-1
    if(l_max .eq. 0) l_max = 1 ! Libray has problems with l_max=0
    m_min = 0
    m_max = size(Qlm,2)-1
    if(m_max > l_max) l_max=m_max

    allocate(Leg%I_LM%F(0:l_max,0:m_max))
    call Legendre( I_LM=Leg%I_LM )
    Qlm(1:size(Qlm,1),:) = Leg%I_LM%F(0:size(Qlm,1)-1,:)
    deallocate(Leg%I_LM%F)
    deallocate(x)
  end subroutine calculate_Qlm_array

  ! Interface functions: get single value at a time (inefficient!)
  function calculate_Plm(l,m,xi) result(R)
    integer, intent(in) :: l
    integer, intent(in) :: m
    double precision, intent(in) :: xi
    double precision :: R
    double precision, dimension(:,:), allocatable :: Plm

    allocate(Plm(0:l,0:m))
    call calculate_Plm_array(Plm,xi)
    R=Plm(l,m)
    deallocate(Plm)
  end function calculate_Plm

  function calculate_Qlm(l,m,xi) result(I)
    integer, intent(in) :: l
    integer, intent(in) :: m
    double precision, intent(in) :: xi
    double precision :: I
    double precision, dimension(:,:), allocatable :: Qlm

    allocate(Qlm(0:l,0:m))
    call calculate_Qlm_array(Qlm,xi)
    I=Qlm(l,m)
    deallocate(Qlm)
  end function calculate_Qlm

  ! C interfaces: calculate arrays
  subroutine calc_Plm_arr(R,lmax,mmax,xi) bind(C,name='calc_Plm_arr')
    real(kind=c_double) :: R((lmax+1)*(mmax+1))
    integer(kind=c_int), value :: lmax
    integer(kind=c_int), value :: mmax
    real(kind=c_double), value :: xi

    double precision, dimension(:,:), allocatable :: Plm
    integer :: ll, mm
    allocate(Plm(0:lmax,0:mmax))
    call calculate_Plm_array(Plm,xi)

    do ll=0,lmax
       do mm=0,mmax
          R(ll*(mmax+1)+mm+1)=Plm(ll,mm)
       end do
    end do
    deallocate(Plm)
  end subroutine calc_Plm_arr

  subroutine calc_Qlm_arr(I,lmax,mmax,xi) bind(C,name='calc_Qlm_arr')
    real(kind=c_double) :: I((lmax+1)*(mmax+1))
    integer(kind=c_int), value :: lmax
    integer(kind=c_int), value :: mmax
    real(kind=c_double), value :: xi

    double precision, dimension (:,:), allocatable :: Qlm
    integer :: ll, mm
    allocate(Qlm(0:lmax,0:mmax))
    call calculate_Qlm_array(Qlm,xi)

    do ll=0,lmax
       do mm=0,mmax
          I(ll*(mmax+1)+mm+1)=Qlm(ll,mm)
       end do
    end do
    deallocate(Qlm)
  end subroutine calc_Qlm_arr

  ! C interfaces: calculate values
  function calc_Plm_val(l,m,xi) result(R) bind(C,name='calc_Plm_val')
    integer(kind=c_int), value :: l
    integer(kind=c_int), value :: m
    real(kind=c_double), value :: xi
    real(kind=c_double) :: R

    R=calculate_Plm(l,m,xi)
  end function calc_Plm_val

  function calc_Qlm_val(l,m,xi) result(I) bind(C,name='calc_Qlm_val')
    integer(kind=c_int), value :: l
    integer(kind=c_int), value :: m
    real(kind=c_double), value :: xi
    real(kind=c_double) :: I

    I=calculate_Qlm(l,m,xi)
  end function calc_Qlm_val

end module legendre_wrapper_m
