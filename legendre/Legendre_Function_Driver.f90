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

!deck Legendre_Function_Driver
!**begin prologue     DRIVER
!**date written       060711   (yymmdd)
!**revision date      yymmdd   (yymmdd)
!**keywords           
!**author             schneider, barry (nsf)
!**source
!**purpose            test code to compute the regular and/or irregular associated Legendre
!***                  functions for integer values of the degree and order and any real argument.
!**description        see CPC paper or type 'yes' at prompt.
!**                   
!**references
!**routines called        
!                       ----                    --------
!
!**end prologue       DRIVER
  PROGRAM Legendre_Function_Driver
  USE accuracy
  USE input_output
  USE Associated_Legendre_Functions
  USE Lentz_Thompson
  IMPLICIT NONE
!                                                                                                  
  TYPE (CF_Legendre)                              :: CFL
  CHARACTER(LEN=8)                                :: itoc
  CHARACTER(LEN=16)                               :: ans
  INTEGER                                         :: i
  INTEGER                                         :: len
  namelist / input_data / title, l_max, m_min, m_max, n_points, upper, lower, &
                          Directive, input_values, normalized, Control,       &
                          Print_Functions, Print_Wronskian, Print_Factors,    &
                          Print_Norms, eps, R, recur, test_wron, smallest, biggest
!
!  Get the input and output file numbers which appear in the 
!  Module input_output.f90
!
!  Open the input and output files
!
  write(6,*) '                  Lets begin the calculation'
  write(6,*) '  If you wish (do not wish) some information, type yes(no)'
  write(6,*) '  First time users should definitely type yes'
  read(5,*) ans
  IF ( ans .eq. 'yes' ) THEN
       Call Info
  ELSE IF (ans .eq. 'no' ) THEN
       OPEN(inp,file='Input_Leg',status='old')
       OPEN(iout,file='Output_Leg',status='unknown')
!
!      If you do wish to take the default values, read the data namelist.
!
       READ(inp,*) ans      
       IF (ans .eq. 'input data' ) THEN
           READ(inp,nml=input_data)
       END IF
!
       write(iout,1)
       len=len_trim(title)
       write(iout,2) title(1:len)
       write(iout,1)
       write(iout,3) l_max, m_min, m_max, n_points, Directive, Control, &
                     normalized, eps, recur, test_wron
!
!
!       If you simply want to read in arbitrary values of the argument set input_values
!       to .true.  Otherwise read in an upper and lower value, a step and a number of
!       points and the code will generate the arguments.  Note that some compilers do not
!       like allocated variables to appear in namelist statements even though the 
!       allocation is done before the variable is read in.  You might have to fix that.
!       The intel compiler is fine with it.
!
        IF (input_values) THEN
            ALLOCATE(x(1:n_points)) 
            READ(inp,*) x(1:n_points)
            upper = x(1)
            lower = x(1)
            DO i = 2, n_points
               upper = max(upper,x(i))
               lower = min(lower,x(i))
           END DO
        ELSE
           step = (upper - lower ) / n_points
           n_points = n_points + int_one
           ALLOCATE(x(1:n_points)) 
           x(1) = lower
           DO i = 2, n_points
              x(i) = x(i-int_one) + step
           END DO
        END IF

!
!        Allocate some space for storage of often used variables
!
!
!       Here we have the option of using either the Miller algorithm (A) or the continued
!       fraction approach (B).
!
!
!       Print the arguments.
!
        title='Grid'
        Call Print_Matrix(x,iout,title=title)
        IF ( recur .eq. 'Miller' ) THEN
             Leg%D%A%Dir=recur
        ELSE IF ( recur .eq. 'Wronskian' ) THEN   
             Leg%D%B%Dir=recur
        END IF
!
!       Calculate either regular Legendre only, irregular Legendre only or both.
!
        IF ( Directive .eq. 'regular') THEN
             ALLOCATE(Leg%R_LM%F(0:l_max,0:m_max))
        ELSE IF ( Directive .eq. 'irregular') THEN
             ALLOCATE(Leg%I_LM%F(0:l_max,0:m_max))
        ELSE IF( Directive .eq. 'both') THEN
             ALLOCATE(Leg%R_LM%F(0:l_max,0:m_max), Leg%I_LM%F(0:l_max,0:m_max))
        END IF
        IF (Leg%D%B%Dir .eq. 'Wronskian' ) THEN
            ALLOCATE(Leg%R_L%F(0:l_max))
        END IF
!
!       Do the calculation
!
        IF ( Directive .eq. 'regular') THEN
             Call Legendre( R_LM=Leg%R_LM, normalized=normalized )
             DEALLOCATE(Leg%R_LM%F)
        ELSE IF ( Directive .eq. 'irregular') THEN
             Call Legendre( I_LM=Leg%I_LM )
             DEALLOCATE(Leg%I_LM%F)
        ELSE IF( Directive .eq. 'both') THEN
             IF ( normalized .eqv. .false. ) THEN
                  Call Legendre( R_LM=Leg%R_LM, I_LM=Leg%I_LM  )
             ELSE
                  Call Legendre( R_LM=Leg%R_LM, I_LM=Leg%I_LM, normalized=normalized )
             END IF
             DEALLOCATE(Leg%R_LM%F, Leg%I_LM%F)
        END IF
        IF (Leg%D%B%Dir .eq. 'Wronskian' ) THEN
            DEALLOCATE(Leg%R_L%F)
        END IF
!        DEALLOCATE( Factor )
        CLOSE(inp)
        CLOSE(iout)
  ELSE
        stop
  END IF
1 Format('           **************************************************************' &
         '****************')
2 Format(15x,'Begin Test Calculation = ',a80)
3 Format(/,10x,'Maximum L                             = ',i6,1x,                     &
               'Minimum M                             = ',i6,1x,                     &
               'Maximum M                             = ',i6,                        &
         /,10x 'Number of Points                      = ',i6,1x,                     &
               'Type Functions Computed        = ',a10,1x,                           &
         /,10x,'Type Calculation               = ',a24,1x,                           &
               'Normalize Regular Functions on (-1,1) = ',l1,1x,                     &
         /,10x,'Continued Fraction Convergence = ',1pe15.8,1x,                       &
               'Backward Recurrence Method            = ',a24,1x,                    &
        /,10x, 'Test Wronskian                 = ',l1) 
  stop
END PROGRAM  Legendre_Function_Driver
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!deck Info
!**begin prologue     Info
!**date written       060711   (yymmdd)
!**revision date      yymmdd   (yymmdd)
!**keywords           
!**author             schneider, barry (nsf)
!**source
!**purpose            Info
!**description        
!**                   
!**references
!**routines called        
!                       ----                    --------
!
!**end prologue       Info
  Subroutine Info
  USE Associated_Legendre_Functions
  CHARACTER(LEN=16)                               :: YN
  write(6,*)  'First a caution.  This code was written using 4 byte integers'
  write(6,*)  'and 8 byte real numbers.'
  write(6,*)  'Consequently, for large values of l and m one may obtain'
  write(6,*)  'one may obtain overflows resulting in NaN or infinity.'
  write(6,*)  'The code will not abort but the values are meaningless.'
  write(6,*)  'It is difficult to predict precisely for what values of l, m and x'
  write(6,*)  'this will happen but the output tells the story.'
  write(6,*)
  write(6,*)  '                          To continue type yes. To quite type no'
  write(6,*)
  read(5,*) YN
  IF ( YN .eq. 'no') THEN
       Call exit
  END IF
  write(6,*)  'Compute P_lm(x) and Q_lm(x) for all x'                                 
  write(6,*)  'For P_lm and Q_lm with x=(-1,1) upward recursion is stable.'
  write(6,*)  'For P_lm for abs(x) > 1 upward recursion is also stable'
  write(6,*)  'For Q_lm for abs(x) > 1 upward recursion is unstable and it is necessary to use'
  write(6,*)  'downward recursion.  It is straightforward to obtain the starting values for'
  write(6,*)  'forward recursion.  For downward recursion, we have developed two approaches;'
  write(6,*)  'one is a modification of Millers method where the starting value is obtained'
  write(6,*)  'from a continued fraction representation of the ratio of the last two Q_lm.'
  write(6,*)  'The other approach uses the same continued fraction but combines it with the'
  write(6,*)  'known wronskian of P_lm and Q_lm to obtain exact values of the last two Q_lm.'
  write(6,*)  'The first method requires a renormalization of the Q_lm after the downward'
  write(6,*)  'recursion while the second method, first advanced by Segura and Gil'
  write(6,*)  'is self contained.  The Segura and Gil approach has the disadvantage that it'
  write(6,*)  'is necessary to compute P_lm to obtain the wronskian.  '
  write(6,*)  'If P_lm is also needed by the user, this is not a disadvantage.'
  write(6,*)  'Both approaches produce accurate results.'
  write(6,*)
  write(6,*)  '                          To continue type yes. To quite type no'
  write(6,*)
  read(5,*) YN
  IF ( YN .eq. 'no') THEN
       Call exit
  END IF
  write(6,*)  'Note that the continued fraction does converge slowly when the argument'
  write(6,*)  'is near 1.'
  write(6,*)
  write(6,*)   '              Define'
  write(6,*)   '         s_fac = 1 if abs(z) <= 1 s_fac = -1 if abs(z) > 1'
  write(6,*)   '         y = sqrt( s_fac * z )'
  write(6,*)   '                 Initial Values'
  write(6,*)   '         G_LM is either P_LM or Q_LM'    
  write(6,*)   '         P_MM = - s_fac * ( 2*M - 1) *  sqrt ( s_fac * ( 1 - x* x ) )'
  write(6,*)   '                                     * P_(M-1)(M-1)'
  write(6,*)   '         Q_00 = .5 * ln ( abs( ( z + 1) /( z - 1))'
  write(6,*)   '         Q_10 = z * Q_00 - 1'
  write(6,*)   '         Q_01 = - 1 / sqrt ( s_fac * ( 1 - z * z ) )'
  write(6,*)   '         Q_11 = - s_fac * sqrt ( s_fac * ( 1 - z * z ) ) * ( Q_00 + z / ( 1 - z * z ) )'
  write(6,*)   '                 Recurances'
  write(6,*)   
  write(6,*)  '( L + 1 - M) G_(L+1)M = ( 2*L + 1) z G_LM - ( L + M ) G_(L-1)M'
  write(6,*)  ' is the recursion used to step up or down in l depending on whether'
  write(6,*)  ' you begin with the first or last two members of the sequence'
  write(6,*)  ' To step up in M use'
  write(6, *) 'G_LM = 2*M z G_L(M-1)/ y - s_fac * ( L + M ) * ( L - M + 1 ) * G_L(M-2)'
  write(6,*)
  write(6,*)  
  write(6,*)  '                          To continue type yes. To quite type no'
  write(6,*)
  read(5,*) YN
  IF ( YN .eq. 'no') THEN
       Call exit
  END IF
  write(6,*) '                 Input Variables'
  write(6,*) 'l_max = maximum l  m_min = minimum m, m_max = maximum m'
  write(6,*)  '              n_points = number of points'
  write(6,*) 'upper = highest value of argument lower = lowest value of argument'
  write(6,*) 'Directive = regular,irregular or both input_values = .true.(Generate from data)'
  write(6,*) 'normalized = .true. ( .false. ) (Normalize or not if the functions if on cut)' 
  write(6,*) 'eps = convergence criterion for continued fraction'
  write(6,*) 'recur = Miller or Wronskian x = read in values of argument'
  write(6,*) 'Many variables have default values so that the users will obtain'                                 
  write(6,*) 'output even when they are novices on the use of the code'                                 
  write(6,*) '                  Default Values'
  write(6,*) 'l_max = ', l_max,' m_max = ',m_max,' n_points = ',n_points
  write(6,*) 'lower = ',lower,' upper = ',upper
  write(6,*) 'Directive = ',Directive
  write(6,*) 'Control = ',Control 
  write(6,*) 'normalized = ',normalized,' eps = ',eps
  write(6,*) 'recur = ',recur
  write(6,*) '                     Here is a sample input file'
  write(6,*) '&input_data'
  write(6,*) 'title="test_legendre", l_max=50, m_max=5, Print=.true.'
  write(6,*) 'n_points=1, input_values=.true., upper=1.0, Control="compute_functions",'
  write(6,*) 'lower=-1.0, directive="irregular", normalized=.true.,' 
  write(6,*) 'recur="Wronskian", eps=1.d-15, R = 1.4 /'
  write(6,*) '1.0001 /'
  End Subroutine Info
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

