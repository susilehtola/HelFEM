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

MODULE Associated_Legendre_Functions
!***begin prologue     Associated_Legendre_Functions
!***date written       021231   (yymmdd)
!***revision date               (yymmdd)                                                  
!***keywords           associated legendre functions                                         
!***author             schneider, b. i.(nsf)                                                 
!***source                                                                                    
!***purpose            Compute P_lm(x) and Q_lm(x) for all x                                 
!***description        See subroutine Info in driver codes for description.
!***references
!***routines called
!***end prologue       Associated_Legendre_Functions
!
!                          Needed Modules
!
  USE Auxilliary_Subroutines
  USE Prolate_Functions
  USE Lentz_Thompson
  IMPLICIT NONE
                                                                                                  
                           INTERFACE Legendre
             MODULE PROCEDURE Legendre                              
                       END INTERFACE Legendre
!                                                                                                  
                           INTERFACE Legendre_Recursion
             MODULE PROCEDURE Upward_Regular_Legendre_Recursion_L,                         &
                              Upward_Regular_Legendre_Recursion_LM,                        &
                              Upward_Irregular_Legendre_Recursion_LM,                      &
                              Downward_Irregular_Legendre_Recursion_LM_A,                  &  
                              Downward_Irregular_Legendre_Recursion_LM_B
                       END INTERFACE Legendre_Recursion
!
                           INTERFACE Initialize
             MODULE PROCEDURE Initialize_Regular_L,                                        &
                              Initialize_Regular_LM,                                       &
                              Initialize_Irregular_L,                                      &
                              Initialize_Irregular_LM                              
                       END INTERFACE Initialize
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!               
                             CONTAINS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*deck Legendre_Functions 
!***begin prologue     Legendre_Functions  
!***date written       091101   (yymmdd)
!***revision date      yymmdd   (yymmdd)
!***keywords           legendre functions
!***author             schneider, barry (lanl)
!***source             
!***purpose            legendre functions
!***description        calculation of P_LM(x) functions using various recursion
!***                   relations.  
!***
!***references         The needed relations can be found at the following two references;
!***                   1. http://en.wikipedia.org/wiki/Associated_Legendre_function
!***                   2. Abramowitz and Stegun, Handbook of Mathematical Functions
!***                           UNITED STATES DEPARTMENT OF COMMERCE
!***                                       With
!***                   Formulas, Graphs, and Mathematical Tables
!***                   Edited by Milton Abramowitz and Irene A. Stegun
!***                   National Bureau of Standards, Applied Mathematics Series - 55
!***                   Issued June 1964, Tenth Printing, December 1972, with corrections
!***                   For sale by the Superintendent of Documents, U.S. Government Printing Office 
!***                   Washington, D.C. 20402 - Price $11.35 domestic postpaid, or $10.50 GPO Bookstore 
!***
!***                   Some comments.  The recursion relationships when the argument is inside or 
!***                   ouside the cut [-1,1] are slightly different.  Reference 2. does not contain 
!***                   all of the recurrances needed.  This is noted in the subroutines.
!                      P_LM(z)[Q_LM(z)] are the regular[irregular] associated legendre 
!***                   functions.  z are the real values of the argument.  Between - 1 and + 1 upward
!                      recursion can be used for both the regular and irregular function.
!                      For other values of z upward recursion is fine for the regular
!                      function but downward recusion must be used for the irregular function.
!***routines called
!***end prologue       Legendre_Functions
      Subroutine Legendre ( R_LM, I_LM, normalized ) 
      IMPLICIT NONE
      TYPE(Reg_LM), OPTIONAL                         :: R_LM
      TYPE(Irreg_LM), OPTIONAL                       :: I_LM
      TYPE(Up)                                       :: U
      TYPE(Down)                                     :: D
      TYPE(Down_A)                                   :: A
      TYPE(Down_B)                                   :: B
      INTEGER                                        ::  i
      TYPE(Normalization), Dimension(:), ALLOCATABLE ::  nrmlm
      INTEGER                                        ::  j
      INTEGER                                        ::  lrow=1
      INTEGER                                        ::  lcol=1
      CHARACTER(LEN=5)                               ::  itoc
      CHARACTER(LEN=16)                              ::  fptoc
      LOGICAL, OPTIONAL                              ::  normalized      
!
!
!----------------------------------------------------------------------c
!
!
!----------------------------------------------------------------------c
!
!----------------------------------------------------------------------c
!
!         Set print labels
!
  ALLOCATE(col_label(int_zero:m_max))
  DO i=int_zero,m_max
     col_label(i) = 'm = '//itoc(i)
  END DO
  ALLOCATE(row_label(int_zero:l_max))
  DO i=int_zero,l_max
     row_label(i) = 'l = '//itoc(i)
  END DO
!
  ALLOCATE(y(int_one:int_twenty))
  ALLOCATE(nrmlm(0:m_max))   
!  IF ( PRESENT(normalized) == .true. ) THEN          
  Call Normalization_Factors(nrmlm)   !    Legendre normalization
!
! END IF
  DO i = int_one, n_points
!----------------------------------------------------------------------c
!
!    The definition and calculation of Legendre functions depends if the
!    argument is ( -1.,1.) or outside that range.  Using s_fac allows a uniform
!    treatment.
     arg = x(i)
     s_fac = int_one
     IF ( abs(arg) > one ) THEN
          s_fac = - int_one
     END IF
     y(int_one) = one - arg * arg
     y(int_two) =  s_fac * y(int_one)
     y(int_three) = sqrt ( y(int_two) )   
     y(int_four) = y(int_three)   
     y(int_five) = arg * arg
     y(int_six) = y(int_five) * arg
     y(int_seven) = y(int_six) * arg
     y(int_eight) = y(int_seven) * arg 
     y(int_nine) = y(int_eight) * arg 
     y(int_ten) = y(int_nine) * arg 
     y(int_eleven) = y(int_ten) * arg 
     y(int_twelve) = y(int_eleven) * arg 
     IF ( PRESENT(R_LM) ) THEN
!
          IF ( abs(arg) <= one) THEN          
               IF ( PRESENT(normalized) ) THEN          
                    Call Legendre_Recursion ( Leg%R_LM, normalized=normalized )
                    IF (Print_Functions) THEN
                        write(iout,1) arg
                        title='Normalized Regular Associated Legendre Functions'
                        write(iout,2) title
                        Call Print_Matrix(Leg%R_LM%F, l_max + int_one, m_max + int_one, iout, frmt='e',          &
                                          collab=col_label, rowlab=row_label )
                    END IF
               ELSE
                    Call Legendre_Recursion ( Leg%R_LM )
                    IF (Print_Functions) THEN
                        write(iout,1) arg
                        title='Unnormalized Regular Associated Legendre Functions'
                        write(iout,2) title
                        Call Print_Matrix(Leg%R_LM%F, l_max + int_one, m_max + int_one, iout, frmt='e',          &
                                          collab=col_label, rowlab=row_label )
                    END IF
               END IF
          ELSE
               Call Legendre_Recursion ( Leg%R_LM )
               IF (Print_Functions) THEN
                   write(iout,1) arg
                   title='Unnormalized Regular Associated Legendre Functions'
                   write(iout,2) title
                   Call Print_Matrix(Leg%R_LM%F, l_max + int_one, m_max + int_one, iout, frmt='e',          &
                                     collab=col_label, rowlab=row_label )
               END IF
          END IF
     END IF
!
!----------------------------------------------------------------------c
!
!----------------------------------------------------------------------c
     IF ( PRESENT(I_LM) ) THEN
!
          IF ( abs(arg) .eq. one) THEN
               write(iout,3)
               return
          END IF
!          log_factor = log ( abs ( ( arg + one )/( arg - one ) ) )
          log_factor = log ( arg + one ) - log( abs(arg - one ) )
!----------------------------------------------------------------------c
!
!         Starting values for Q_LM upward recursion.
!----------------------------------------------------------------------c
!
          IF( abs(arg) < one ) THEN
              Call Legendre_Recursion( Leg%I_LM, Leg%U )
          ELSE
!----------------------------------------------------------------------c
!           Recur downward for m = 0,1 using either Millers algorithm  c
!           or the continued fraction.
!----------------------------------------------------------------------c
!
               IF ( Leg%D%A%Dir .eq. 'Miller' ) THEN
                    Call Legendre_Recursion(  Leg%I_LM, Leg%D, Leg%D%A )
               ELSE IF (Leg%D%B%Dir .eq. 'Wronskian' ) THEN
                    Call Legendre_Recursion(  Leg%I_LM, Leg%D, Leg%D%B )
               END IF
          END IF
          IF (Print_Functions) THEN
              title='Irregular Associated Legendre Functions'
              write(iout,1) arg
              write(iout,2) title
              Call Print_Matrix(Leg%I_LM%F, l_max + int_one, m_max + int_one, iout, frmt='e',   &
                                collab=col_label, rowlab=row_label )
          END IF
     END IF 
     IF ( PRESENT(Normalized) .eqv. .false. ) THEN
          write(iout,4)
          IF ( PRESENT (R_LM) .and. PRESENT(I_LM) .and. test_wron ) THEN
               Call Wronskian ( Leg%R_LM , Leg%I_LM, nrmlm )
          END IF
     ELSE
          write(iout,5)
     END IF
  END DO
  DO m = int_zero, m_max                                                                                              
     DEALLOCATE(nrmlm(m)%leg_fac)
  END DO     
  DEALLOCATE(nrmlm)   
  DEALLOCATE( y )
  DEALLOCATE(col_label)
  DEALLOCATE(row_label)
1 Format(/,25x,'Argument = ',f15.8)
2 Format(/,25x,a48)
3 Format(/,25x,'Cannot Compute Irregular Function for Argument One')
4 Format(/,25x,'Wronskian tested only for un-normalized Legendre functions')
5 Format(/,25x,'Wronskian not tested for normalized Legendre functions')
END SUBROUTINE Legendre
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*deck Initialize_Regular_L 
!***begin prologue     Initialize_Regular_L    
!***date written       091101   (yymmdd)
!***revision date      yymmdd   (yymmdd)
!***keywords           legendre functions
!***author             schneider, barry (lanl)
!***source             
!***purpose            legendre functions
!***description        calculation starting values of regular Legendre functions
!***                   for upward L recursion when only one M value needed.
!***references         none
!                      
!***routines called
!***end prologue       Initialize_Regular_L         
      Subroutine Initialize_Regular_L ( R_L, normalized )
      IMPLICIT NONE
      TYPE ( Reg_L )                              :: R_L
      REAL(idp), DIMENSION(2)                     :: a_lm
      LOGICAL, OPTIONAL                           :: normalized      
!----------------------------------------------------------------------c
!----------------------------------------------------------------------c
!              
!         Starting values for P_LM.
!              
!----------------------------------------------------------------------c
!----------------------------------------------------------------------c
!       Use: 
!              P_MM = - s_fac * ( 2*M - 1) *  sqrt ( s_fac * ( 1 - x* x ) ) * P_(M-1)(M-1)
!       To step up in M after initializing at one.  
!
!
!              Overwrite starting value until the result is obtained.
!
  IF ( PRESENT(normalized) .eqv. .false. ) THEN            
       a_lm(1) = int_one
       Leg%R_L%F(m) = one
       DO l = int_one, m
          Leg%R_L%F(m) = - s_fac * a_lm(1) * y(int_three) * Leg%R_L%F(m)
          a_lm(1) = a_lm(1) + int_two
       END DO
!
!              Calculate the second P term.
!
       IF ( l_max > m ) THEN
!
!             Now calculate:
!                    P_(M+1)M
!
            a_lm(1) = m + m + int_one
            Leg%R_L%F(m+1) = a_lm(1) * arg * Leg%R_L%F(m)
       END IF
  ELSE
       Leg%R_L%F(m) = a_fac
       a_lm(1) = 0
       DO l = int_one, m
          a_lm(2) = ( a_lm(1) + int_three ) / ( a_lm(1) + int_two )
          Leg%R_L%F(m) = - s_fac * sqrt(a_lm(2)) * y(int_three) * Leg%R_L%F(m)
          a_lm(1) = a_lm(1) + int_two
       END DO
!
!
!              Calculate the second P term.
!                    P_(M+1)M
       IF ( l_max > m ) THEN
            a_lm(1) = m + m + int_three
            Leg%R_L%F(m+1) = sqrt(a_lm(1)) * arg * Leg%R_L%F(m)
       END IF
  END IF
END SUBROUTINE Initialize_Regular_L  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*deck Initialize_Regular_LM 
!***begin prologue     Initialize_Regular_LM    
!***date written       091101   (yymmdd)
!***revision date      yymmdd   (yymmdd)
!***keywords           legendre functions
!***author             schneider, barry (lanl)
!***source             
!***purpose            legendre functions
!***description        calculation starting values of regular Legendre functions
!***                   for upward L recursion when multiple M values are needed.
!***references         none
!                      
!***routines called
!***end prologue       Initialize_Regular_LM         
      Subroutine Initialize_Regular_LM ( R_LM, normalized )
      IMPLICIT NONE
      TYPE ( Reg_LM )                             :: R_LM
      REAL(idp), DIMENSION(3)                     :: a_lm
      LOGICAL, OPTIONAL                           :: normalized      

!----------------------------------------------------------------------c
!----------------------------------------------------------------------c
!              
!         Starting values for P_LM.
!              
!----------------------------------------------------------------------c
!
!
!----------------------------------------------------------------------c
!       Use: 
!              P_MM = - s_fac * ( 2*M - 1) *  sqrt ( s_fac * ( 1 - x* x ) ) * P_(M-1)(M-1)
!       To step up in M after initializing at one.  
!
  IF ( PRESENT(normalized) .eqv. .false. ) THEN            

       DO m = int_zero, m_max              
!
!               Initialize first value.
!
          Leg%R_LM%F(m,m) = one
          a_lm(1) = int_one 
          DO l = int_one, m
             Leg%R_LM%F(m,m) = - s_fac * a_lm(1) * y(int_three) * Leg%R_LM%F(m,m)
             a_lm(1) = a_lm(1) + int_two
          END DO
!
!               Calculate second value.
!
          IF (l_max /= m) THEN
!
!          Now calculate:
!                 P_(M+1)M
!
              Leg%R_LM%F(m+int_one,m) = a_lm(1) * arg * Leg%R_LM%F(m,m)
              a_lm(1) = a_lm(1) + int_two
          END IF
       END DO
  ELSE
       DO m = int_zero, m_max              
!
!               Initialize first value.
!
          Leg%R_LM%F(m,m) = a_fac
          a_lm(1) = int_three 
          a_lm(2) = int_two 
          DO l = int_one, m
             a_lm(3) = sqrt(a_lm(1)/a_lm(2))
             Leg%R_LM%F(m,m) = - s_fac * a_lm(3) * y(int_three) * Leg%R_LM%F(m,m)
             a_lm(1) = a_lm(1) + int_two
             a_lm(2) = a_lm(2) + int_two 
          END DO
!
!               Calculate second value.
!
          IF (l_max /= m) THEN
!
!          Now calculate:
!                 P_(M+1)M
!
              a_lm(1) = m + m + int_three
              Leg%R_LM%F(m+int_one,m) = sqrt(a_lm(1)) * arg * Leg%R_LM%F(m,m)
              a_lm(1) = a_lm(1) + int_two
          END IF
       END DO
  END IF
END SUBROUTINE Initialize_Regular_LM  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*deck Initialize_Irregular_L 
!***begin prologue     Initialize_Irregular_L    
!***date written       091101   (yymmdd)
!***revision date      yymmdd   (yymmdd)
!***keywords           legendre functions
!***author             schneider, barry (lanl)
!***source             
!***purpose            legendre functions
!***description        calculation starting values of regular Legendre functions
!***                   for upward L recursion when only one M is needed.
!***references         none
!                      
!***routines called
!***end prologue       Initialize_Irregular_L         
      Subroutine Initialize_Irregular_L ( I_L )
      IMPLICIT NONE
      TYPE ( Irreg_L )                            :: I_L
      REAL(idp)                                   :: I_0
!
!
!         Starting values for Q_LM upward recursion.
!              Use:       
!                    Q_00 = .5 * ln ( abs( ( z + 1) /( z - 1))
!                    Q_10 = z * Q_00 - 1
!                    Q_01 = - 1 / sqrt ( s_fac * ( 1 - z * z ) )
!                    Q_11 = - sqrt ( s_fac * ( 1 - z * z ) ) * ( Q_00 + z / ( 1 - z * z ) )
!
!----------------------------------------------------------------------c
!

    IF ( m .eq. int_zero) THEN
         Leg%I_L%F(int_zero) = half * log_factor
         Leg%I_L%F(int_one) = arg * Leg%I_L%F(int_zero) - one
    END IF
    IF ( m .eq. int_one) THEN
         I_0 = half * log_factor
         Leg%I_L%F(int_zero) =  - one / y(int_three)
         Leg%I_L%F(int_one) =  - s_fac * y(int_three) * ( I_0 + arg / y(int_one) )
    END IF
!    
END SUBROUTINE Initialize_Irregular_L  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*deck Initialize_Irregular_LM 
!***begin prologue     Initialize_Irregular_LM    
!***date written       091101   (yymmdd)
!***revision date      yymmdd   (yymmdd)
!***keywords           legendre functions
!***author             schneider, barry (lanl)
!***source             
!***purpose            legendre functions
!***description        calculation starting values of regular Legendre functions
!***                   for upward L recursion when multiple M values needed.
!***references         none
!                      
!***routines called
!***end prologue       Initialize_Irregular_LM         
      Subroutine Initialize_Irregular_LM ( I_LM )
      IMPLICIT NONE
      TYPE ( Irreg_LM )                           :: I_LM
!
!
!         Starting values for Q_LM upward recursion.
!              Use:       
!                    Q_00 = .5 * ln ( abs( ( z + 1) /( z - 1))
!                    Q_10 = z * Q_00 - 1
!                    Q_01 = - 1 / sqrt ( s_fac * ( 1 - z * z ) )
!                    Q_11 = - sqrt ( s_fac * ( 1 - z * z ) ) * ( Q_00 + z / ( 1 - z * z ) )
!
!----------------------------------------------------------------------c
!
  Leg%I_LM%F(int_zero,int_zero) = half * log_factor
  Leg%I_LM%F(int_one,int_zero) = arg * Leg%I_LM%F(int_zero,int_zero) - one
  IF ( m_max > int_zero) THEN
       Leg%I_LM%F(int_zero,int_one) =  - one / y(int_three)
       Leg%I_LM%F(int_one,int_one) =  - s_fac * y(int_three) * ( Leg%I_LM%F(int_zero,int_zero) + arg / y(int_one) )
  END IF
!
END SUBROUTINE Initialize_Irregular_LM  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*deck Upward_Regular_Legendre_Recursion_L 
!***begin prologue     Upward_Regular_Legendre_Recursion_L   
!***date written       091101   (yymmdd)
!***revision date      yymmdd   (yymmdd)
!***keywords           legendre functions
!***author             schneider, barry (lanl)
!***source             
!***purpose            legendre functions
!***description        calculation of Q_LM(z) functions using forward recursion in L.
!***                   For the un-normalized function the recursion formulat is;
!                      ( L + 1 - M) P_(L+1)M = ( 2*L + 1) z P_LM - ( L + M ) P_(L-1)M
!***                   The Recursion is started with the explicit forms of P_MM and P_(M+1)M
!***                   The forward recursion is stable for both the regular and irregular
!***                   functions as long as abs(z) <= one and L is not huge.  It is also stable
!***                   for the regular functions for abx(z) > one.  It is NOT stable for the
!***                   irregular functions under these conditions and backward recursion is 
!***                   required.  The recursion for the normalized functions substitutes in
!***                                              ^
!***                                  P_LM = N_LM P_LM 
!***                   with N_LM = sqrt ( 2 ( L + M )! / ( 2L + 1) ( L - M )! ). 
!***                   can be used to compute the new coefficients in the recursion.
!***                   They are more complicated after they are computed, the same procedure is used.
!***references         none
!                      
!***routines called
!***end prologue       Upward_Regular_Legendre_Recursion_L         
      Subroutine Upward_Regular_Legendre_Recursion_L ( R_L, normalized)
      IMPLICIT NONE
      TYPE ( Reg_L )                              :: R_L
      INTEGER                                     :: two_m
      REAL(idp), DIMENSION(9)                     :: a_lm
      LOGICAL, OPTIONAL                           :: normalized
!
!
!----------------------------------------------------------------------c
!----------------------------------------------------------------------c
!              
!              Starting values for P_LM.
!              
!----------------------------------------------------------------------c
!----------------------------------------------------------------------c
!
!
  two_m = m + m
  IF ( PRESENT(normalized) .eqv. .false. ) THEN              
       Call Initialize ( Leg%R_L ) 
!----------------------------------------------------------------------c
!----------------------------------------------------------------------c
!              Get the other L values by upward recursion in l
!----------------------------------------------------------------------c
!----------------------------------------------------------------------c
!
!         The upward recursion.  Stable for all values of z
!         This is just the standard l recursion in textbooks.
!
       a_lm(1) = two_m + int_three 
       a_lm(2) = two_m + int_one
       a_lm(3) = int_two
       DO l = m + int_two, l_max
          Leg%R_L%F(l) = ( a_lm(1) * arg * Leg%R_L%F(l - int_one)                &
                                     -                                          &
                           a_lm(2) * Leg%R_L%F(l - int_two) ) / a_lm(3)  
          a_lm(1) = a_lm(1) + int_two
          a_lm(2) = a_lm(2) + int_one
          a_lm(3) = a_lm(3) + int_one
       END DO
  ELSE
       Call Initialize ( Leg%R_L, normalized=normalized )
       a_lm(1) = int_two            ! Starting value for l-m+1 for l =  m + 1
       a_lm(2) = two_m + int_two    ! Starting value for l+m+1 for l =  m + 1
       a_lm(3) = two_m + int_one    ! Starting value for 2*l-1 for l =  m + 1
       a_lm(4) = two_m + int_three  ! Starting value for 2*l+1 for l =  m + 1
       a_lm(5) = two_m + int_five   ! Starting value for 2*l+3 for l =  m + 1
       a_lm(6) = two_m + int_one    ! Starting value for l+m   for l =  m + 1
       a_lm(7) = int_one            ! Starting value for l-m   for l =  m + 1
       DO l = m + int_two, l_max
          a_lm(8) = a_lm(4) * a_lm(5) / ( a_lm(1) * a_lm(2) )
          a_lm(8)  = sqrt ( a_lm(8) )
          a_lm(9)  = a_lm(5) * a_lm(6) * a_lm(7) / ( a_lm(1) * a_lm(2) * a_lm(3) )
          a_lm(9)  = sqrt ( a_lm(9) )
          Leg%R_L%F(l) =   a_lm(8) * arg * Leg%R_L%F(l - int_one)                &
                                     -                                          &
                           a_lm(9) * Leg%R_L%F(l - int_two)  
          a_lm(1) = a_lm(1) + int_one
          a_lm(2) = a_lm(2) + int_one
          a_lm(3) = a_lm(3) + int_two
          a_lm(4) = a_lm(4) + int_two
          a_lm(5) = a_lm(5) + int_two
          a_lm(6) = a_lm(6) + int_one
          a_lm(7) = a_lm(7) + int_one
       END DO 
  END IF
!
END SUBROUTINE Upward_Regular_Legendre_Recursion_L 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*deck Upward_Regular_Legendre_Recursion_LM 
!***begin prologue     Upward_Regular_Legendre_Recursion_LM    
!***date written       091101   (yymmdd)
!***revision date      yymmdd   (yymmdd)
!***keywords           legendre functions
!***author             schneider, barry (lanl)
!***source             
!***purpose            legendre functions
!***description        calculation of Q_LM(z) functions using forward recursion in L.
!***                   ( L + 1 - M) Q_(L+1)M = ( 2*L + 1) z Q_LM - ( L + M ) Q_(L-1)M
!***                   Recursion started with the explicit forms of Q_MM and Q_(M+1)M
!***                   The forward recursion is stable for both the regular and irregular
!***                   functions as long as abs(z) <= one and L is not huge.  It is also stable
!***                   for the regular functions for abx(z) > one.  It is NOT stable for the
!***                   irregular functions under these conditions and backward recursion is 
!***                   required.
!***references         none
!                      
!***routines called
!***end prologue       Upward_Regular_Legendre_Recursion_LM          
      Subroutine Upward_Regular_Legendre_Recursion_LM ( R_LM, normalized )
      IMPLICIT NONE
      TYPE ( Reg_LM )                             :: R_LM
      INTEGER                                     ::  l
      LOGICAL, OPTIONAL                           :: normalized
      INTEGER                                     :: two_m
      REAL(idp), DIMENSION(9)                     :: a_lm
!
!
!----------------------------------------------------------------------c
!----------------------------------------------------------------------c
!              
!              Starting values for P_LM.
!              
!----------------------------------------------------------------------c
!----------------------------------------------------------------------c
!
!
  IF ( PRESENT(normalized) .eqv. .false. ) THEN              
       Call Initialize ( Leg%R_LM ) 
!
!----------------------------------------------------------------------c
!----------------------------------------------------------------------c
!              Get the other L values by upward recursion
!----------------------------------------------------------------------c
!----------------------------------------------------------------------c
!
!
       DO m = int_zero, m_max
          two_m = m + m
!
!         The upward recursion.  Stable for all values of z
!         This is just the standardv reecursion in textbooks.
!
          a_lm(1) = two_m + int_three
          a_lm(2) = two_m + int_one
          a_lm(3) = int_two
          DO l = m + int_two, l_max
             Leg%R_LM%F(l,m) = ( a_lm(1) * arg * Leg%R_LM%F(l - int_one,m)             &
                                            -                                         &
                                 a_lm(2) * Leg%R_LM%F(l - int_two,m) ) / a_lm(3)
             a_lm(1) = a_lm(1) + int_two
             a_lm(2) = a_lm(2) + int_one
             a_lm(3) = a_lm(3) + int_one
          END DO
       END DO
  ELSE
       Call Initialize ( Leg%R_LM, normalized=normalized )
       DO m = int_zero, m_max
          two_m = m + m
          a_lm(1) = int_two            ! Starting value for l-m+1 for l =  m + 2
          a_lm(2) = two_m + int_two    ! Starting value for l+m+1 for l =  m + 2
          a_lm(3) = two_m + int_one    ! Starting value for 2*l-1 for l =  m + 2
          a_lm(4) = two_m + int_three  ! Starting value for 2*l+1 for l =  m + 2
          a_lm(5) = two_m + int_five   ! Starting value for 2*l+3 for l =  m + 2
          a_lm(6) = two_m + int_one    ! Starting value for l+m   for l =  m + 2
          a_lm(7) = int_one            ! Starting value for l-m   for l =  m + 2
          DO l = m + int_two, l_max
             a_lm(8)  = a_lm(4) * a_lm(5) / ( a_lm(1) * a_lm(2) )
             a_lm(8)  = sqrt ( a_lm(8) )
             a_lm(9)  =  a_lm(5) * a_lm(6) * a_lm(7) / ( a_lm(1) * a_lm (2) * a_lm(3) )
             a_lm(9)  = sqrt ( a_lm(9) )
             Leg%R_LM%F(l,m) = a_lm(8) * arg * Leg%R_LM%F(l - int_one,m)            &
                                      -                                            &
                               a_lm(9) * Leg%R_LM%F(l - int_two,m)  
             a_lm(1) = a_lm(1) + int_one
             a_lm(2) = a_lm(2) + int_one
             a_lm(3) = a_lm(3) + int_two
             a_lm(4) = a_lm(4) + int_two
             a_lm(5) = a_lm(5) + int_two
             a_lm(6) = a_lm(6) + int_one
             a_lm(7) = a_lm(7) + int_one
          END DO 
       END DO 
  END IF
END SUBROUTINE Upward_Regular_Legendre_Recursion_LM  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*deck Upward_Irregular_Legendre_Recursion_LM 
!***begin prologue     Upward_Irregular_Legendre_Recursion_LM    
!***date written       091101   (yymmdd)
!***revision date      yymmdd   (yymmdd)
!***keywords           legendre functions
!***author             schneider, barry (lanl)
!***source             
!***purpose            legendre functions
!***description        calculation of Q_LM(z) functions using forward recursion in L.
!***                   This is used when abs(arg) is < one.
!***                   ( L + 1 - M) Q_(L+1)M = ( 2*L + 1) z Q_LM - ( L + M ) Q_(L-1)M
!***                   Recursion started with the explicit forms of Q_MM and Q_(M+1)M
!***                   The forward recursion is stable for both the regular and irregular
!***                   functions as long as abs(z) <= one and L is not huge.  It is also stable
!***                   for the regular functions for abx(z) > one.  It is NOT stable for the
!***                   irregular functions under these conditions and backward recursion is required.
!***references         none
!                      
!***routines called
!***end prologue       Upward_Irregular_Legendre_Recursion_LM          
      Subroutine Upward_Irregular_Legendre_Recursion_LM ( I_LM, U )
      IMPLICIT NONE
      TYPE ( Irreg_LM )                             :: I_LM
      TYPE ( Up )                                   :: U
      INTEGER                                       ::  l
      REAL(idp), DIMENSION(3)                       :: a_lm
!----------------------------------------------------------------------c
!
!         Starting values for Q_00, Q_01, Q_10 and Q_11 
!         for upward recursion.
!----------------------------------------------------------------------c
!
  write(iout,*) 'upward recursion'
  Call Initialize( Leg%I_LM )
!
!----------------------------------------------------------------------c
!         Get other values by upward recursion in L starting with
!         the M=0,1 values and then to all L and M by upward
!         recursion on both variables.
!----------------------------------------------------------------------c
!
!         The upward recursion.  Stable for all values of z <= one
!
!----------------------------------------------------------------------c
!         Step up to get all Q_l0 and Q_l1
!----------------------------------------------------------------------c

  a_lm(1) = int_three 
  a_lm(2) = int_one
  a_lm(3) = int_two      
  DO l = int_two, l_max
     Leg%I_LM%F(l,int_zero) = ( a_lm(1) * arg * Leg%I_LM%F(l - int_one,int_zero)          &
                                                 -                                        &
                                a_lm(2) * Leg%I_LM%F(l - int_two,int_zero) ) / a_lm(3)
     a_lm(1) = a_lm(1) + int_two
     a_lm(2) = a_lm(2) + int_one
     a_lm(3) = a_lm(3) + int_one
  END DO
  IF ( m_max > int_zero ) THEN
!
       a_lm(1) = int_three 
       a_lm(2) = int_two
       a_lm(3) = int_one      
       DO l = int_two, l_max
          Leg%I_LM%F(l,int_one) = ( a_lm(1) * arg * Leg%I_LM%F(l - int_one,int_one)      &
                                                   -                                     &
                                    l * Leg%I_LM%F(l - int_two,int_one) ) / ( l - int_one )
          a_lm(1) = a_lm(1) + int_two
          a_lm(2) = a_lm(2) + int_one
          a_lm(3) = a_lm(3) + int_one
       END DO
  END IF
!
!----------------------------------------------------------------------c
!             Now for each L value, step up in M if needed.
!----------------------------------------------------------------------c
  IF ( m_max > int_one ) THEN
       DO l = int_zero, l_max
!
!                     The upward recursion in m
!
          a_lm(1) = - int_two 
          a_lm(2) = l + int_one
          a_lm(3) = l      
          DO m = int_two, m_max
             Leg%I_LM%F(l,m) = ( - m - m + int_two ) * arg * Leg%I_LM%F(l,m-int_one) / y(3)    &
                                                     -                                         &
                                  s_fac * ( l + m - int_one) * ( l - m + int_two) * Leg%I_LM%F(l,m-int_two)
             a_lm(1) = a_lm(1) - int_two
             a_lm(2) = a_lm(2) + int_one
             a_lm(3) = a_lm(3) - int_one
          END DO
       END DO
  END IF
END SUBROUTINE Upward_Irregular_Legendre_Recursion_LM  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*deck Downward_Irregular_Legendre_Recursion_LM_A 
!***begin prologue     Downward_Irregular_Legendre_Recursion_LM_A    
!***date written       091101   (yymmdd)
!***revision date      yymmdd   (yymmdd)
!***keywords           legendre functions
!***author             schneider, barry (lanl)
!***source             
!***purpose            legendre functions
!***description        calculation of Q_LM(z) functions using backward recursion
!***                   and the modified Miller algorithm.
!***                   This is used when abs(arg) is > one.
!***                   ( L + 1 - M) T_LM = ( 2*L + 3) z T_(L+1)M - ( L - M + 2 ) T_(L+2)M
!***                   Starting at a large value of L set the last value to zero and the
!***                   next to last to one.  The recur downward which is the stable direction.
!***                   The T's are proportional to the desired Q functions.  
!***                   The proportionality constant is determined by the known value of Q00.
!***                   This allows us to compute the Q's for m=0. The process is repeated 
!***                   for Q_01
!***references         none
!                      
!***routines called
!***end prologue       Downward_Irregular_Legendre_Recursion_LM_A         
      Subroutine Downward_Irregular_Legendre_Recursion_LM_A ( I_LM, D, A )
      IMPLICIT NONE
      TYPE ( Irreg_LM )                             :: I_LM
      TYPE(CF_Legendre)                             :: CFL
      TYPE ( Down )                                 :: D
      TYPE ( Down_A )                               :: A
      REAL(idp)                                     :: I_2 
      REAL(idp)                                     :: I_1 
      REAL(idp)                                     :: I_0 
      INTEGER                                       :: l
      REAL(idp), DIMENSION(3)                       :: a_lm
      write(iout,*) 'downward recursion'
!
!     Compute continued fraction Q_L_max / Q_(L_max-1) for m=0
!
      m = int_zero
      Call Continued_Fraction_Legendre(CFL,I_2,arg,l_max,m)    
      Leg%I_LM%F(l_max,m) = I_2 * smallest
      Leg%I_LM%F(l_max-int_one,m) = smallest
!
!     Downward recursion for m = 0
!
      a_lm(1) = m - l_max 
      a_lm(2) = l_max + l_max - int_one
      a_lm(3) = l_max + m - int_one
      DO l = l_max-2, int_zero, - int_one
         Leg%I_LM%F(l,m) = ( a_lm(2) * arg * Leg%I_LM%F(l+int_one,m) + a_lm(1) * Leg%I_LM%F(l+int_two,m) ) / a_lm(3)
         a_lm(1) = a_lm(1) + int_one
         a_lm(2) = a_lm(2) - int_two
         a_lm(3) = a_lm(3) - int_one
      END DO
!
!     Renormalize using known value of first member,
!
      scale_factor =  half * log_factor  / Leg%I_LM%F(int_zero,m)
      Leg%I_LM%F(int_zero:l_max,m) = scale_factor * Leg%I_LM%F(int_zero:l_max,m)
      IF ( m_max > int_zero) THEN
!
!         Downward recursion for m = 1
!
!
!         Compute continued fraction Q_L_max / Q_(L_max-1) for m=1
!
          m = int_one
          Call Continued_Fraction_Legendre(CFL,I_2,arg,l_max,m)
          Leg%I_LM%F(l_max,m) = I_2 * smallest
          Leg%I_LM%F(l_max-int_one,m) = smallest
!
!         Downward recursion 
!
          a_lm(1) = m - l_max 
          a_lm(2) = l_max + l_max - int_one
          a_lm(3) = l_max + m - int_one
          DO l = l_max-2, int_zero, - int_one
             Leg%I_LM%F(l,m) = ( a_lm(2) * arg * Leg%I_LM%F(l+int_one,m) + a_lm(1) * Leg%I_LM%F(l+int_two,m) ) / a_lm(3)
             a_lm(1) = a_lm(1) + int_one
             a_lm(2) = a_lm(2) - int_two
             a_lm(3) = a_lm(3) - int_one
          END DO
!
!         Renormalize using known value of first member.
!
          scale_factor = ( - one / y(int_three) ) / Leg%I_LM%F(int_zero,m)
          Leg%I_LM%F(int_zero:l_max,m) = scale_factor * Leg%I_LM%F(int_zero:l_max,m)
      END IF
!
!     For each l value, step up in m
!
      DO l = int_zero, l_max
         a_lm(1) = - int_two
         a_lm(2) = l + int_one
         a_lm(3) = l
!
!        The upward recursion in m
!
         DO m = int_two, m_max
            Leg%I_LM%F(l,m) = a_lm(1) * arg * Leg%I_LM%F(l,m - int_one) / y(3)       &
                                           -                                        &
                              s_fac * a_lm(2) * a_lm(3) * Leg%I_LM%F(l, m - int_two)
            a_lm(1) = a_lm(1) - int_two
            a_lm(2) = a_lm(2) + int_one
            a_lm(3) = a_lm(3) - int_one
         END DO
      END DO
!1 Format(/,10x,'Argument = ',e15.8,1x,'Top L = ',i5)
!2 Format(/,25x,a48)
END SUBROUTINE Downward_Irregular_Legendre_Recursion_LM_A  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!*deck Downward_Irregular_Legendre_Recursion_LM_B 
!***begin prologue     Downward_Irregular_Legendre_Recursion_LM_B    
!***date written       091101   (yymmdd)
!***revision date      yymmdd   (yymmdd)
!***keywords           legendre functions
!***author             schneider, barry (lanl)
!***source             
!***purpose            legendre functions
!***description        calculation of Q_LM(z) functions using a novel idea of Segura and Gil;
!***                   This is used when abs(arg) is < one.
!***                   1. continued fraction for Q_L0/Q_(L-1)0 and Q_L1/Q_(L-1)1
!***                   2. upward recursion for P_L0 and P_L1
!***                   3. the wronskian  P_L0 * Q_(L-1)0 - P_(L-1)0 * Q_L0  = 1 / L 
!***                      wronskian  P_L1 * Q_(L-1)1 - P_(L-1)1 * Q_L1  =  - 1 / L 
!***                      to compute the two  highest values of Q_L0 and Q_L1.  
!***                   4. downward recursion for all Q_L0 and Q_L1.  
!***                   5. upward recursion in M for all QLM.
!***references         Gil and Segura CPC 1998
!                      
!***routines called
!***end prologue       Downward_Irregular_Legendre_Recursion_LM_B         
      Subroutine Downward_Irregular_Legendre_Recursion_LM_B ( I_LM, D, B ) 
      IMPLICIT NONE
      TYPE ( Irreg_LM )                             :: I_LM
      TYPE ( Reg_L )                                :: R_L
      TYPE ( Down )                                 :: D
      TYPE ( Down_B )                               :: B
      TYPE(CF_Legendre)                             :: CFL
      REAL(idp)                                     :: I_2 
      REAL(idp)                                     :: I_1 
      REAL(idp)                                     :: I_0 
      REAL(idp)                                     :: cf 
      INTEGER                                       :: l
      REAL(idp), DIMENSION(3)                       :: a_lm
      INTEGER                                       :: n_2
      INTEGER                                       :: n_3
      write(iout,*) 'downward recursion'
      m = int_zero
!
!                      Recur up for P_L0
!
      Call initialize(Leg%R_L)
      Call Legendre_Recursion ( Leg%R_L )
!
!                      Get continued fraction
!
      Call Continued_Fraction_Legendre(CFL,cf,arg,l_max,m)       
      Leg%I_LM%F(l_max-int_one,m) = 1.d0                            &
                                         /                          &
                      ( l_max * ( Leg%R_L%F(l_max) - Leg%R_L%F(l_max-int_one) * cf ) )      
      Leg%I_LM%F(l_max,m) = cf * Leg%I_LM%F(l_max-int_one,m)            
!
!     Downward recursion for m = 0
!
      a_lm(1) = l_max + l_max - int_one
      a_lm(2) = l_max 
      a_lm(3) = l_max - int_one
      DO l = l_max, int_two, - int_one
         Leg%I_LM%F(l-int_two,m) = ( a_lm(1) * arg * Leg%I_LM%F(l-int_one,m)         &
                                                      -                             &
                                            a_lm(2) * Leg%I_LM%F(l,m) ) / a_lm(3) 
         a_lm(1) = a_lm(1) - int_two
         a_lm(2) = a_lm(2) - int_one
         a_lm(3) = a_lm(3) - int_one
      END DO
      IF ( m_max > int_zero) THEN
           m = int_one
!
!                      Recur up for P_L1
!
           Call initialize(Leg%R_L)
           Call Legendre_Recursion ( Leg%R_L )
!
!                      Get continued fraction
!
           Call Continued_Fraction_Legendre(CFL,cf,arg,l_max,m)       
           Leg%I_LM%F(l_max-int_one,m) = - l_max / ( Leg%R_L%F(l_max) - Leg%R_L%F(l_max-int_one) * cf )   
           Leg%I_LM%F(l_max,m) = cf * Leg%I_LM%F(l_max-int_one,m)            
!           write(iout,*) Leg%I_LM%F(l_max,m), Leg%I_LM%F(l_max-int_one,m)
!
!                      Downward recursion for m = 1
!
           a_lm(1) = l_max + l_max - int_one
           a_lm(2) = l_max - int_one
           a_lm(3) = l_max  
           DO l = l_max, int_two, - int_one
              Leg%I_LM%F(l-int_two,m) = ( a_lm(1) * arg * Leg%I_LM%F(l-int_one,m)         &
                                                    -                                    &
                                          a_lm(2) * Leg%I_LM%F(l,m) ) / a_lm(3) 
              a_lm(1) = a_lm(1) - int_two
              a_lm(2) = a_lm(2) - int_one
              a_lm(3) = a_lm(3) - int_one
           END DO
      END IF

!
!     For each l value, step up in m
!
      DO l = int_zero, l_max
         a_lm(1) = - int_two
         a_lm(2) = l + int_one
         a_lm(3) = l
!
!        The upward recursion in m
!
         DO m = int_two, m_max
            Leg%I_LM%F(l,m) = a_lm(1) * arg * Leg%I_LM%F(l,m - int_one) / y(3)       &
                                           -                                        &
                              s_fac * a_lm(2) * a_lm(3) * Leg%I_LM%F(l, m - int_two)
            a_lm(1) = a_lm(1) - int_two
            a_lm(2) = a_lm(2) + int_one
            a_lm(3) = a_lm(3) - int_one
         END DO
      END DO
1 Format(/,5x,'m = ',i3,1x,'Continued Fraction = ',d15.8)
2 Format(/,5x,'Starting Values of Legendre Functions')
3 Format(/,5x,5e15.8)
END SUBROUTINE Downward_Irregular_Legendre_Recursion_LM_B  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
END MODULE Associated_Legendre_Functions
