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

MODULE Auxilliary_Subroutines
!***begin prologue     Special_Functions
!***date written       021231   (yymmdd)
!***revision date               (yymmdd) 
!***keywords           Auxilliary_Subroutines                                                              
!***author             schneider, b. i.(nsf)
!***source             
!***purpose                                                                                 
!***description        
!***                   
!***                   
!***                   
!***                   
!***                   
!***                                                                                                  
!
!***references
!***routines called
!***end prologue                                                  
!
  USE accuracy
  USE Data_Module
  USE Special_Functions
  IMPLICIT NONE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                 Contains
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*deck Factorials 
!***begin prologue     Factorials    
!***date written       091101   (yymmdd)
!***revision date      yymmdd   (yymmdd)
!***keywords           legendre functions
!***author             schneider, barry (lanl)
!***source             
!***purpose            Factorials 
!***description        Factorials
!***                     
!***references         none
!                      
!***routines called
!***end prologue       Factorials         
      Subroutine Factorials 
      IMPLICIT NONE
      INTEGER                  :: i
      Factor(0) = int_one
      Factor(int_one) = int_one
      DO i = int_two, l_max + m_max
         Factor(i) = i * Factor( i - int_one )
      END DO
END SUBROUTINE Factorials 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*deck Wronskian 
!***begin prologue     Wronskian     
!***date written       091101   (yymmdd)
!***revision date      yymmdd   (yymmdd)
!***keywords           legendre functions
!***author             schneider, barry (lanl)
!***source             
!***purpose            legendre functions
!***description        
!***references         Gil and Segura CPC 1998
!                      
!***routines called
!***end prologue       Wronskian         
      Subroutine Wronskian ( R_LM, I_LM, nrmlm ) 
      IMPLICIT NONE
      TYPE ( Reg_LM )                               :: R_LM
      TYPE ( Irreg_LM )                             :: I_LM
      TYPE(Normalization), Dimension(0:m_max)       :: nrmlm                                                      
      REAL(idp)                                     :: wr_calc
      REAL(idp)                                     :: max_err
      REAL(idp)                                     :: diff
!
      m_sign = 1
      DO m = 0, m_max
         max_err = zero
         DO l = m + int_one, l_max
            wron = m_sign / ( nrmlm(m)%leg_fac(l) * ( l + m ) )
            wr_calc = Leg%R_LM%F(l,m) * Leg%I_LM%F(l-int_one,m)            &
                                      -                                    &
                      Leg%R_LM%F(l-int_one,m) * Leg%I_LM%F(l,m)
            diff = ( wron - wr_calc )/ wron
            max_err = max(max_err,abs(diff))
            IF (Print_Wronskian) THEN
                write(iout,1) m, l, wron, wr_calc
            END IF
         END DO
         write (iout,2) m, max_err
         IF (abs(arg) > one ) THEN
             m_sign = - m_sign
         END IF
      END DO
!                 
1 Format(/,5x,'m = ',i3,1x,'l = ',i3,/,10x,                                &
              'Exact Wronskian = ',1pe20.12,1x,'Computed Wronskian = ',1pe20.12)
2 Format(/,5x,'m = ',i3,1x,'Maximum Relative Error in Wronskian for all Computed l Values = ',1pe20.12)
END SUBROUTINE Wronskian
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*deck Normalization_Factors                                                                           
!***begin prologue     Normalization_Factors                                                           
!***date written       091101   (yymmdd)                                                               
!***revision date      yymmdd   (yymmdd)                                                               
!***keywords           legendre functions                                                              
!***author             schneider, barry (lanl)                                                         
!***source                                                                                             
!***purpose            Compute normalization of P_LM on (-1,1)                          
!***description        N_Factorial                                                                     
!***                                                                                                   
!***references         none                                                                            
!                                                                                                      
!***routines called                                                                                    
!***end prologue       Normaliztion_Factors                                                            
      Subroutine Normalization_Factors(nrmlm)                                                          
      IMPLICIT NONE                                                                                    
      TYPE(Normalization), Dimension(0:m_max) :: nrmlm                                                      
      INTEGER                                 :: m                                                          
      INTEGER                                 :: l                                                          
      INTEGER                                 :: m_plus                                                          
      REAL(idp), DIMENSION(2)                 :: a_lm                                                          
      REAL(idp)                               :: inv_sqrt2                                                          
      CHARACTER(LEN=5)                        :: itoc
      DO m = int_zero, m_max                                                                                              
         ALLOCATE(nrmlm(m)%leg_fac(m:l_max))  ! ( l-m )! / ( l + m )!
      END DO     
      nrmlm(0)%leg_fac(0:l_max) = one
!
!             Recur up in m using previous values of (m,l)
! 
      DO m = int_zero, m_max - int_one
         m_plus = m + int_one
         a_lm(1) = m_plus + m_plus
         a_lm(2) = int_one
         DO l = m_plus, l_max
            nrmlm(m_plus)%leg_fac(l) = nrmlm(m)%leg_fac(l) / ( a_lm(1) * a_lm(2) )
            a_lm(1) = a_lm(1) + int_one
            a_lm(2) = a_lm(2) + int_one
         END DO
      END DO
      IF (Print_Factors) THEN
          DO m = int_zero, m_max 
             title='Normalization Factors m = '//itoc(m)
             write(iout,*) title
             Write(iout,*) nrmlm(m)%leg_fac(m:l_max)
          END DO
      END IF
!
!             Compute the normalization factors
!
      inv_sqrt2 = 1.d0 / sqrt2
      DO m = int_zero, m_max
         ALLOCATE(nrmlm(m)%norm(m:l_max))
         a_lm(1) = m + m + int_one
         DO l = m, l_max
            nrmlm(m)%norm(l) = sqrt ( a_lm(1) * nrmlm(m)%leg_fac(l) ) * inv_sqrt2
            a_lm(1) = a_lm(1) + int_two
         END DO 
      END DO
      IF (Print_Norms) THEN
          DO m = int_zero, m_max
             title='Normalization Constants m = '//itoc(m)
             write(iout,*) title
             Write(iout,*) nrmlm(m)%norm(m:l_max)
          END DO
      END IF
END SUBROUTINE Normalization_Factors 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*deck Print_Norm_Factors 
!***begin prologue     Print_Norm_Factors      
!***date written       091101   (yymmdd)
!***revision date      yymmdd   (yymmdd)
!***keywords           legendre functions
!***author             schneider, barry (lanl)
!***source             
!***purpose            legendre functions
!***description        
!***references         Gil and Segura CPC 1998
!                      
!***routines called
!***end prologue       Print_Norm_Factors          
      Subroutine Print_Norm_Factors(nrmlm) 
      IMPLICIT NONE
      TYPE(Normalization), Dimension(0:m_max)       :: nrmlm                                                      
      DO l = 0, m_max
         write(iout,*) 'The m value = ',l
         write(iout,*) 'from l = ',l,' to l =',l_max
         write(iout,*) nrmlm(l)%leg_fac(0:l_max)
      END DO    
END SUBROUTINE Print_Norm_Factors 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*deck Renormalize 
!***begin prologue     Renormalize   
!***date written       091101   (yymmdd)
!***revision date      yymmdd   (yymmdd)
!***keywords           legendre functions
!***author             schneider, barry (lanl)
!***source             
!***purpose            legendre functions
!***description        Normalized regular Legendre functions from their un-normalized values.
!***                   This is only used on the cut (-1,+1).
!***                     
!***references         none
!                      
!***routines called
!***end prologue       Renormalizen        
      Subroutine Renormalize ( F_lm, nrmlm )
      IMPLICIT NONE
      REAL(idp), DIMENSION(0:l_max,int_zero:m_max)     ::  F_lm
      TYPE(Normalization), Dimension(0:m_max)          :: nrmlm 
      DO m = int_zero, m_max
         DO l = m , l_max
            F_lm(l,m) = nrmlm(m)%norm(l) * F_lm(l,m)
         END DO
      END DO
END SUBROUTINE Renormalize 
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
END MODULE Auxilliary_Subroutines
