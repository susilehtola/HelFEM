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

MODULE Lentz_Thompson
!***begin prologue     Lentz_Thompson
!***date written       021231   (yymmdd)
!***revision date               (yymmdd)                                                                                            
!***keywords           Lentz_Thompson                                                              
!***author             schneider, b. i.(nsf)                                                                                        
!***source                                                                                                                    
!***purpose            Compute continued fractions using Lentz-Thompson algoritm                                                                     
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
!***end prologue       Lentz_Thompson                                                                               
!
!                          Needed Modules
!
  USE accuracy
  USE Data_Module
  USE input_output
  USE Matrix_Print
  USE Special_Functions
  IMPLICIT NONE
!
!
                           INTERFACE Continued_Fractions
             MODULE PROCEDURE Continued_Fraction_Legendre
                       END INTERFACE Continued_Fractions                                                                    
!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                                      
                             CONTAINS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*deck Continued_Fraction_Legendre
!***begin prologue     Continued_Fraction_Legendre  
!***date written       100206   (yymmdd)
!***revision date      yymmdd   (yymmdd)
!***keywords           continued fractions
!***author             schneider, barry (NSF)
!***source             
!***purpose            Compute continued fraction
!***description        
!***                   
!***references         none
!***routines called
!***end prologue       Continued_Fraction_Legendre  
  Subroutine Continued_Fraction_Legendre(CFL,f,x,nu,mu) 
  IMPLICIT NONE
  TYPE(CF_Legendre)                 :: CFL
  REAL(idp)                         :: x 
  INTEGER                           :: nu 
  INTEGER                           :: mu 
  INTEGER                           :: n_0 
  REAL(idp)                         :: a 
  REAL(idp)                         :: b 
  REAL(idp)                         :: f        
  REAL(idp)                         :: C        
  REAL(idp)                         :: D        
  REAL(idp)                         :: Del        
  REAL(idp)                         :: test        
  INTEGER                           :: count         
  INTEGER                           :: iwrite         
!
  f = smallest
  C = f
  D = zero
  test = one
  a = one 
  n_0 = nu
  b = zero
!  Write(iout,1) f, C, D, test, a, b, n_0
  count = 0
!  iwrite = 0
  DO While (test > eps )              
     count = count + int_one
!     iwrite = iwrite + int_one
     b = ( ( n_0 + n_0 + one ) * x ) / ( n_0 + mu )
     D = b + a * D
     IF ( D .eq. zero ) THEN
          D = smallest
     END IF
     C = b + a / C
     IF ( C .eq. zero ) THEN
          C = smallest
     END IF
     D = 1.d0 / D
     Del = C * D
     f = f * Del
     test = abs ( Del - one )
     a = - ( n_0 - mu + one ) / ( n_0 + mu )
     n_0 = n_0 + 1
!     IF ( iwrite == 50 ) THEN
!          iwrite = 0
!          Write(iout,2) count, f, C, D, test, a, b, n_0
!     END IF
  END DO
  Write(iout,3) nu, mu, count, test, f 
1 Format(5x, 'Initial Values',/,10x,                                    &
         'f    = ',d20.12,1x,'C = ',d15.8,1x,'D = ',d15.8,/,10x,        &
         'test = ',d15.8,1x,'a  = ',d15.8,1x,'b = ',d15.8,1x,'n_0 = ',i10)
2 Format(5x, 'Loop Values',5x,'Count = ',i5/,10x,                       &
         'f    = ',d20.12,1x,'C = ',d15.8,1x,'D = ',d15.8,/,10x,        &
         'test = ',d15.8,1x,'a  = ',d15.8,1x,'b = ',d15.8,1x,'n_0 = ',i10)
3 Format(1x,'l = ',i3,1x,'m = ',i3,1x,'Iteration Count = ',i5,          &
         /,1x,'Convergence = ',1pe19.11,1x,'Final Value of Continued Fraction = ',1pe19.11)
END SUBROUTINE Continued_Fraction_Legendre
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
END MODULE Lentz_Thompson
