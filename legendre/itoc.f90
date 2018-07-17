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

!function itoc.f90	5.1  11/6/94
      function itoc(num)
!***begin prologue     itoc
!***date written       850601  (yymmdd)
!***revision date      yymmdd  (yymmdd)
!***keywords           character, integer, conversion
!***author             martin, richard (lanl)
!***source             @(#)itoc.f	5.1   11/6/94
!***purpose            converts an integer into a string.
!***description
!                      itoc is a character function used as:
!                      string=itoc(num)
!                      num  the integer to convert.
!
!                      itoc will work for integers between 10**-15 and 10**16.
!
!***references
!***routines called    (none)
!***end prologue       itoc
      implicit integer(a-z)
      character*(*) itoc
      character digits*10,k*1,str*16
      data digits/'0123456789'/
      save digits
      maxsiz=len(itoc)
      str=' '
      n=iabs(num)
      i=0
!     generate digits.
   10 i=i+1
         d=mod(n,10)
         str(i:i)=digits(d+1:d+1)
         n=n/10
         if(n.gt.0.and.i.lt.maxsiz) goto 10
!     generate sign.
      if(num.lt.0.and.i.lt.maxsiz) then
         i=i+1
         str(i:i)='-'
      endif
!
!     now flip it around.
      halfi=i/2
      do 20 j=1,halfi
         k=str(j:j)
         str(j:j)=str(i+1-j:i+1-j)
         str(i+1-j:i+1-j)=k
   20 continue
      itoc=str
      return
      end
