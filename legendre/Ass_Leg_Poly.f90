!    This software was developed by employees of the National Institute of 
!    Standards and Technology (NIST), an agency of the Federal Government and 
!    is being made available as a public service. Pursuant to title 17 United 
!    States Code Section 105, works of NIST employees are not subject to 
!    copyright protection in the United States.  This software may be subject 
!    to foreign copyright.  Permission in the United States and in foreign 
!    countries, to the extent that NIST may hold copyright, to use, copy, 
!    modify, create derivative works, and distribute this software and its 
!    documentation without fee is hereby granted on a non-exclusive basis, 
!    provided that this notice and disclaimer of warranty appears in all copies.
!    
!    THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, 
!    EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, 
!    ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY 
!    IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, 
!    AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION 
!    WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE 
!    ERROR FREE.  IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, 
!    INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR 
!    CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY 
!    CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, 
!    CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY 
!    PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED 
!    FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR 
!    SERVICES PROVIDED HEREUNDER.
!
!

MODULE Ass_Leg_Poly
!***begin prologue     Ass_Leg_Poly
!***date written       021231   (yymmdd)
!***revision date               (yymmdd)                                                  
!***keywords           associated legendre functions                                         
!***author             schneider, b. i.(nsf)                                                 
!***source                                                                                    
!***purpose            Compute P_lm(x) and Q_lm(x) for all x                                 
!***description        See subroutine Info in driver codes for description.
!***references
!***routines called
!***end prologue       Ass_Leg_Poly
!
!                          Needed Modules
!
  USE accuracy
  USE input_output
  USE Matrix_Print
  IMPLICIT NONE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!               
                             CONTAINS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!deck P_LM
!**begin prologue     P_LM
!**date written       880721   (yymmdd)
!**revision date      yymmdd   (yymmdd)
!**keywords           legend, link 2702, legendre functions
!**author             schneider, barry (lanl)
!**source             m2702
!**purpose            Regular Legendre functions
!**description        calculation of p(l,m) functions
!**references         none
!                      plm are the legendre functions l=m to l=lmax    
!                      x are the values of cos(theta)
!**routines called
!**end prologue       P_LM
      subroutine P_LM (plm,x,m,l_max)
      IMPLICIT NONE
      REAL(idp), DIMENSION(m:l_max) :: plm
      REAL(idp)                     :: x
      REAL(idp)                     :: p_start
      REAL(idp)                     :: arg
      INTEGER                       :: m
      INTEGER                       :: l_max
      INTEGER                       :: n_1
      INTEGER                       :: n_2
      INTEGER                       :: n_3
      INTEGER                       :: i
!----------------------------------------------------------------------
!           start recursion with plm(m,m) and plm(m+1,m)               
!                      and recur upward                                
!----------------------------------------------------------------------
  arg =sqrt( 1.d0 - x*x )
  plm(m) = 1
  n_1 = 1
  DO i = 1, m
     plm(m) = - arg * plm(m)
     n_1 = n_1 + 2
  END DO
  plm(m+1) = ( m + m + 1 ) * x
  n_1 = 3
  n_2 = m + m + 3
  n_3 = m + m + 1
  DO i = m + 1 , l_max
     plm(i+1) = ( n_2 * arg  - n_3 / plm(i) ) / n_1
     n_1 = n_1 + 1
     n_2 = n_2 + 2
     n_3 = n_3 + 1
  END DO
  plm(m+1) = plm(m+1) * plm(m)
  DO i = m+1, l_max
     plm(i+1) = plm(i+1) * plm(i)
  END DO  
  END Subroutine P_LM
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
END MODULE Ass_Leg_Poly
