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

MODULE Prolate_Spheroidal_Coulomb
!***begin prologue     Prolate_Spheroidal_Coulomb
!***date written       021231   (yymmdd)
!***revision date               (yymmdd) 
!***keywords           Prolate_Spheroidal_Coulomb                                                              
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
  USE Auxilliary_Subroutines
  USE Prolate_Functions
  IMPLICIT NONE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                 Contains
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*deck Prolate_Coulomb_Interaction                                                                           
!***begin prologue     Prolate_Coulomb_Interaction                                                                                 
!***date written       091101   (yymmdd)                                                               
!***revision date      yymmdd   (yymmdd)                                                               
!***keywords           legendre functions                                                              
!***author             schneider, barry (lanl)                                                         
!***source                                                                                             
!***purpose            Compute coulomb interaction as harmonic expansion in prolate spheroidal coordinates                          
!***description                                                                             
!***                                                                                                   
!***references         none                                                                            
!                                                                                                      
!***routines called                                                                                    
!***end prologue      Prolate_Coulomb_Interaction 
  Subroutine Prolate_Coulomb_Interaction(nrmlm)                                                          
  IMPLICIT NONE                                                                                    
  TYPE(Normalization), Dimension(0:m_max) :: nrmlm                                                      
  INTEGER                                 :: m                                                          
  INTEGER                                 :: l                                                          
  INTEGER                                 :: m_plus                                                          
  varphi_diff = varphi(1) - varphi(2)
  csum_real = 0.0_idp
  csum_imag = 0.0_idp
  DO l =  0, l_max
     dl21  = dble(l + l + 1)
     DO m = -l, l
        vardm  = dble(m) * varphi_diff
        mabs = ABS(m)
        meo  = MOD(mabs,2)
        facm =  1.0_idp
        IF (meo .eq. 1) THEN
            facm = -1.0_idp
        END IF
        temp = nrmlm(mabs)%leg_fac(l)
        temp = temp*temp
        temp = facm*dl21*temp
        temp = temp * Leg%R_LM%Xi%F_Small(l,mabs) * Leg%I_LM%Xi%F_Large(l,mabs)   &
                                                  *                               &
                      Leg%R_LM%Eta%F_1(l,mabs)    * Leg%R_LM%Eta%F_2(l,mabs)     
        ctemp_real = temp * COS(vardm) 
        ctemp_imag = temp * SIN(vardm)
        csum_real = csum_real + ctemp_real
        csum_imag = csum_imag + ctemp_imag
     END DO
  END DO
  csum_real = two * csum_real * a_p(3)
  csum_imag = two * csum_imag * a_p(3)
  Write(iout,'(1x,a,es22.15)') '1/r_12 (Neuman) real = ',csum_real
  Write(iout,'(1x,a,es22.15)') '1/r_12 (Neuman) imag = ',csum_imag
  Write(iout,'(1x,a,es22.15)') '1/r_12 (direct)      = ',r_12_invs
END SUBROUTINE Prolate_Coulomb_Interaction
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
END MODULE Prolate_Spheroidal_Coulomb
