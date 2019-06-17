function ezmti5(i1,i2,numtype,acr);#,gthreswidth,sthreswidth); [blob blobvel ccframevel] =
  #numtype is 6 for file name numbers of 6 long, 8 for 8 long
  #acr is % in pixels of total image in width that is the size of error frame and track start
  #**dxy3>Fdiff>blur>blobs***************************************************************************
  #Easy Moving Target Indicator, Black and White image inputs only
  #*****************************************************************************
  #i1 and i2 = first frame to last frame to be processed
  #*****************************************************************************
  #uint16 image req'd for .gif video  uint8 req'd for imshow()
  #Try to keep all images as uint8 in main function code, allows consistent uint8 in / out of subfunctions
      tic;
  track=0;#0=acquisition mode 1=track mode
  framesizenlarge=4;#increase in sub frame around blob when comparing fdiff to origimage blobs to certify a moving blob
  nofdiffblobs=0;
  vfnum=0;
  bkframes=5;
  tgtareas(:,1)=[1;1;1;0;1];#the col# is the areanumber, rows:iavg; javg; pwavg; numberoftgthitsinthisarea; ageofarea
  tgtareaval=1;#=max(tgtareas,2); 1column in tgtareas, init target area value
  #errmaxsize=uint16(0.25*size(pic1,1));ACTUAL CODE AROUND LINE 38 AFTER PIC1 IS READIN
  #*****************************************************************************
  for n1 = (i1+1):i2; #start at 2nd img,  start(i1) to end (i2)
    #RESET CRITICAL VARIABLES
    bcnt=0;
    tcount=0;
    #GET IMAGES
    if numtype == 6
      pic2=getimage6(n1-1);#get 2nd image (past time)
      pic1=getimage6(n1);#get most recent image, default is uint16 image
    endif
    if numtype == 8
      pic2=getimage8(n1-1);#get 2nd image (past time)
      pic1=getimage8(n1);#get most recent image, default is uint16 image
    endif
    if numtype == 5
      pic2=getimage5(n1-1);#get 2nd image (past time)
      pic1=getimage5(n1);#get most recent image, default is uint16 image
    endif
%    #Acquire the Track frame, tf, (subset of total image), so that we are not processing on entire image
%    if track==1
%      tfstartrow=ptrow-(ptwidthrow/2);
%      if tfstartrow<1
%        tfstartrow=1;
%      endif
%      tfstoprow=ptrow+(ptwidthrow/2);
%      if tfstoprow>size(pic1,1)
%        tfstoprow=size(pic1,1);
%      endif
%      tfstartcol=ptcol-(ptwidthcol/2);
%      if tfstartcol<1
%        tfstartcol=1;
%      endif
%      tfstopcol=ptcol+(ptwidthcol/2);
%      if tfstopcol>size(pic1,2)
%        tfstopcol=size(pic1,2);
%      endif
%      trackpic=pic1(tfstartrow:tfstoprow,tfstartcol:tfstopcol);#trackpic is a subframe of img
%      gradthres=.6;
%      [dx3,dy3,dxy3] = grad3(uint8(trackpic));#take Gradient and use max pos xx% and max negXX% of gradient result as target
%      dx3pos=uint8((dx3-(gradthres*(max(max(dx3))))).*250);#the .5 can be adjustable in future
%      negtemp=dx3.*(-1);
%      dx3neg=uint8((negtemp-(gradthres*(max(max(negtemp))))).*250);
%      dx3img=or(dx3pos,dx3neg);
%      dy3pos=uint8((dy3-(gradthres*(max(max(dy3))))).*250);#the .5 can be adjustable in future
%      negtemp=dy3.*(-1);
%      dy3neg=uint8((negtemp-(gradthres*(max(max(negtemp))))).*250);
%      dy3img=or(dy3pos,dy3neg);
%      dxy3pos=uint8((dxy3-(gradthres*(max(max(dxy3))))).*250);#the .5 can be adjustable in future
%      negtemp=dxy3.*(-1);
%      dxy3neg=uint8((negtemp-(gradthres*(max(max(negtemp))))).*250);
%      dxy3img=or(dxy3pos,dxy3neg);
%      img=or(dx3img,dy3img,dxy3img);
%      ssrows=sum(img,1);#compress rows into a vector, provides blob center column
%      sscols=sum(img,2);#blob center row
%      for a=1:size(ssrows,2)
%        if ssrows(a)>0
%          rowstart=a;
%          break
%        endif
%      endfor
%      for a=[size(ssrows,2):-1:1]
%        if ssrows(a)>0
%          rowstop=a;
%          break
%        endif
%      endfor
%      for a=1:size(sscols,1)
%        if sscols(a)>0
%          colstart=a;
%          break
%        endif
%      endfor
%      for a=[size(sscols,1):-1:1]
%        if sscols(a)>0
%          colstop=a;
%          break
%        endif
%      endfor
%      rowcntr=tfstartrow+rowstart+((rowstart-rowstop)/2);#trackpic start + trackdetection start + half way between track start stop
%      colcntr=tfstartcol+colstart+((colstart-colstop)/2);
%      ptwidthrow=(rowstart-rowstop)*1.5;#multiplier can be adjustable in future, self adjusting, learning
%      ptwidthcol=(colstart-colstop)*1.5;
%      ptwidthavg=(ptwidthrow+ptwidthcol)/2;
%      targetimg = pointdrawsquare(pic1,rowcntr,colcntr,ptwidthavg,250);
%      targetimg = pointdrawplus(targetimg,rowcntr,colcntr,ptwidthavg,250); 
%      vfnum=vfnum+1;
%      picvid1(:,:,1,vfnum)=targetimg;#make video frame
%      continue#skip remaining loop get next frame
%    endif
    #Set Max Target Error Zone   
    errmaxsize=uint16(acr*size(pic1,1));
    #DXY3 Grad
    [a,b,pic2dxy3] = grad3(uint8(pic2));
    [a,b,pic1dxy3] = grad3(uint8(pic1));
    #FRAME DIFF
    df1=uint8(uint8(pic2dxy3)-uint8(pic1dxy3));#this can be improved upon greatly, disregards neg gradients
    thres=.75*(max(max(df1)));#normalized to 0.xx max
    df1=uint8((df1-thres)*254);#16bit ceiling push all bits
    #BLOB FINDING IN PROCESSED IMAGE
    if any(any(df1)) #only process if there are positive df1 pixels existing
      [frmstartrow frmstoprow frmstartcol frmstopcol] = findimagedges(df1);#get new image subframe smaller than entire frame to make ccblobfinding faster
      if frmstartrow>1#ccblobfinder does not handle edges well, so we add this addition one pixel borders
        frmstartrow=frmstartrow-1;
      endif
      if frmstoprow<size(df1,1)
        frmstoprow=frmstoprow+1;
      endif
      if frmstartcol>1
        frmstartcol=frmstartcol-1;
      endif
      if frmstopcol<size(df1,2)
        frmstopcol=frmstopcol+1;
      endif
      bfindimg=df1(frmstartrow:frmstoprow,frmstartcol:frmstopcol);
%      figure; imshow(bfindimg); 
      blob2=0;
      [blob2 ccframe2] = ccblobfinder(bfindimg);
      blobsfound=size(blob2,3);
      for e = [1:blobsfound]
        blb=blob2(:,:,e);
        if any(any(blb))
          [blbstartrow blbstoprow blbstartcol blbstopcol] = findimagedges(blb);
          blbheight=blbstoprow-blbstartrow;
          blbwidth=blbstopcol-blbstartcol;
          blbarea=blbheight*blbwidth;
          blbmidrow=blbstartrow+((blbstoprow-blbstartrow)/2);
          blbmidcol=blbstartcol+((blbstopcol-blbstartcol)/2);
          totblbcntrrow=frmstartrow+blbmidrow;#row
          totblbcntrcol=frmstartcol+blbmidcol;#row
          bcnt=bcnt+1;
          tdandblobs(bcnt,:)=[totblbcntrcol totblbcntrrow blbheight blbwidth blbarea];#[andrank blob# imgcenterrow imgcentercol] #record each blob's rank size and location
        endif
      endfor
    endif 
    #INCREMENT FRAME COUNT 'VFNUM'
    vfnum=vfnum+1;
    #GET TARGET BLOB FOUND
    if bcnt==1
      itgt=tdandblobs(1,2);
      jtgt=tdandblobs(1,1);
      pwtgt=((tdandblobs(1,3))+(tdandblobs(1,4)))/2;
    endif
    if bcnt>1 #Pick the larget blob
      sstdandblobs=sortrows(tdandblobs,5);#resort the blobs for area column
      itgt=sstdandblobs(1,2);
      jtgt=sstdandblobs(1,1);
      pwtgt=((sstdandblobs(1,3))+(sstdandblobs(1,4)))/2;
    endif 
    #PERSISTENCE AND TARGET MARKING
       #tgtareas(:,1)=[1;1;1;0;1];#the col# is the areanumber, rows:iavg(row); javg(col); pwavg; numberoftgthitsinthisarea; ageofarea
    targetimg=pic2;#Persistence Process Loop, latest frame is 'process', prior frame is static
    for a=1:size(tgtareas,2)
      ierr=abs(itgt-(tgtareas(1,a)));
      jerr=abs(jtgt-(tgtareas(2,a)));
      tgtareas(5,a)=tgtareas(5,a)+1;#inrement the age of each target area by 1
      if and(ierr<errmaxsize,jerr<errmaxsize)
        newiavg=(((tgtareas(1,a))*tgtareas(4,a))+itgt)/(1+tgtareas(4,a));#newavg=(logged average * logged occurances)+ (new point) / 1+logged occurances
        newjavg=(((tgtareas(2,a))*tgtareas(4,a))+jtgt)/(1+tgtareas(4,a));
        tgtareas(1,a)=newiavg;
        tgtareas(2,a)=newjavg;
        tgtareas(3,tgtareaval)=pwtgt;#update pw
        tgtareas(4,a)=tgtareas(4,a)+1;
        tgtareas(5,a)=tgtareas(5,a)-1;#decrement the age of each target area by 1; offsets increment above due to this new addition
      else#errors are too large, so make a new area
        if a==size(tgtareas,2);#loop at last cycle, make a new target area to log target averages inside
          tgtareaval=tgtareaval+1;#increment # of target areas
          tgtareas(1,tgtareaval)=itgt;
          tgtareas(2,tgtareaval)=jtgt;
          tgtareas(3,tgtareaval)=pwtgt;
          tgtareas(4,tgtareaval)=1;
          tgtareas(5,tgtareaval)=1;#new target area gets age=1
        endif
      endif  
    endfor
    for a=1:size(tgtareas,2)#Persis Zone Death Loop
      if tgtareas(5,a)>bkframes#get rid of old areas
        tgtareas(:,a)=[];#remove target areas that are older than bkframes
        tgtareaval=tgtareaval-1;
      endif
      break#jump out of this for loop or it errors on the next 'a' due to removal of area
    endfor
    for a=1:size(tgtareas,2)#Persis Zone Marking Loop, maybe old ones died and new ones born
      if tgtareas(4,a)==5
        track=1;
        ptrow=tgtareas(1,a);#set target center row to be passed to Tracker
        ptcol=tgtareas(2,a);#set target center col to be passed to Tracker
        ptwidthrow=errmaxsize;#set target row width window to be passed to Tracker
        ptwidthcol=errmaxsize;#set target col width window to be passed to Tracker
      endif
      maxcount=sum(tgtareas,2);#sum the cols of tgtareas to get total count of occurances
      maxcount=maxcount(4);#occurance hits in 4rth rows
      if maxcount>bkframes#max count value is number of bkframes
        maxcount=bkframes;
      endif
      pixhlt=uint8(125+(125*((tgtareas(4,a))/maxcount)));
      ia=tgtareas(1,a);
      ja=tgtareas(2,a);
      pwa=errmaxsize;#tgtareas(3,a);
      targetimg = pointdrawsquare(targetimg,ia,ja,pwa,pixhlt);
      targetimg = pointfullplus(targetimg,ia,ja,pixhlt);
    endfor
    #SAVE FRAME TO VIDEO FILE
    picvid1(:,:,1,vfnum)=targetimg;#make video frame
  endfor
  imwrite(picvid1,["ezmti5-pic2-persis5-dummy.gif"],"DelayTime",0.2);#" num2str(imagethresdivs) ".gif"],"DelayTime",0.2);
  toc;
endfunction
#*******************************************************************************
#MAIN FUNCTION END
#*******************************************************************************

#Getimage6
function pic = getimage6(n1);#5 total digits in file name, ie: 00001.png
    file1=[num2str(n1) ".png"]; #starts at 1 less than entry to enable a 0 start
    if (n1 < 10)         file1=["00000" file1];
      elseif (n1<100)    file1=["0000" file1];
      elseif (n1<1000)   file1=["000" file1];
      elseif (n1<10000)  file1=["00" file1];
      else               file1=["0" file1];
    endif
    pic=imread(file1);
    pic=pic(:,:,1);#raw picture frame2im
endfunction

#Getimage5
function pic = getimage5(n1);#5 total digits in file name, ie: 00001.png
    file1=[num2str(n1) ".png"]; #starts at 1 less than entry to enable a 0 start
    if (n1 < 10)        file1=["0000" file1];
      elseif (n1<100)   file1=["000" file1];
      elseif (n1<1000)  file1=["00" file1];
      else              file1=["0" file1];
    endif
    pic=imread(file1);
    pic=pic(:,:,1);#raw picture frame2im
endfunction

#Getimage8
function pic = getimage8(n1); #8 total digits in file name, ie: 00000001.png
    file1=[num2str(n1) ".png"]; #starts at 1 less than entry to enable a 0 start
    if (n1 < 10)          file1=["0000000" file1];
      elseif (n1<100)     file1=["000000" file1];
      elseif (n1<1000)    file1=["00000" file1];
      elseif (n1<10000)   file1=["0000" file1];
      elseif (n1<100000)  file1=["000" file1];
      elseif (n1<1000000) file1=["00" file1];
      else file1=["0" file1];
    endif
    pic=imread(file1);
    pic=pic(:,:,1);#raw picture frame2im
endfunction

#Blur1
function blrpic = blur1(pic);#Blur using simple primary axis kernel or any 
  [rsize,csize]=size(pic);
  initmaxval=max(max(pic));#max value in pic, used later to rescale back to uint8 image
%  kernel=[1 0 1 0 1 #semi-Gaussian kernel that can be applied by row then clms, real Gaussian is valid this way rather than full n x m matrix application
%          0 4 4 4 0
%          1 4 9 4 1
%          0 4 4 4 0
%          1 0 1 0 1];#55
  kernel=[2 1 1 1];#as above, from center going positive on prime axis only [10 10 7 5 3 1 1]  / [2 1]
  layers=(size(kernel,2))-1;
  weight=0;#used to divide the summed kernel 
  sumpic=0;
  for n=1:layers
    horzpos(:,:,n)=(kernel(n+1)).*shift(pic,n,1);
    horzneg(:,:,n)=(kernel(n+1)).*shift(pic,-1*n,1);
    vertpos(:,:,n)=(kernel(n+1)).*shift(pic,n,2);
    vertneg(:,:,n)=(kernel(n+1)).*shift(pic,-1*n,2);
    diag1pos(:,:,n)=(kernel(n+1)).*shift(horzpos(:,:,n),n,2);
    diag1neg(:,:,n)=(kernel(n+1)).*shift(horzneg(:,:,n),-1*n,2);
    diag2pos(:,:,n)=(kernel(n+1)).*shift(horzpos(:,:,n),-1*n,2);
    diag2neg(:,:,n)=(kernel(n+1)).*shift(horzneg(:,:,n),n,2);
    sumpic=sumpic+horzpos(:,:,n)+horzneg(:,:,n)+vertpos(:,:,n)+vertneg(:,:,n)+diag1pos(:,:,n)+diag1neg(:,:,n)+diag2pos(:,:,n)+diag2neg(:,:,n);
    weight=weight+(8*kernel(n+1));#addup the weighting; there are 8 kernel multipliers above: horzpos, horzneg, vertpos,...
  endfor
    sumpic=sumpic+((kernel(1)).*pic);#add in the center value kernel multiplied image here
    weight=weight+kernel(1);#center value
    blrpic=sumpic./weight;
    scalingratio=initmaxval/(max(max(blrpic)));
    blrpic=uint8(scalingratio.*blrpic);
%      blrpic=blrpic-pic;#extra edge detection tests
%      blrpic=blrpic+abs(min(min(blrpic)));#shift up out of negative values
endfunction

#thresfrommax
function threspic1 = thresfrommax(image,threswidth);#image thresholding of 'threswidth' from max value in image and down
    threspic1=image+(255-max(max(image)));#shift pixel value high to set thres ceiling
    threspic1=uint8(threspic1-(255-threswidth));#shift pixel down to original level and further for floor
    #threspic1=threspic1.*(240/max(max(threspic1)));#normalize the image for uint8
endfunction

#threshilo
function threspic1 = threshilo(image,threshi,threslo);#image thresholding using a given hi and lo threshold values
    threspic1=image+(255-threshi);#shift pixel value high to set thres ceiling
    threspic1=uint8(threspic1-(255-threshi+threslo));#shift pixel down to original level and further for floor
    #threspic1=threspic1.*(240/max(max(threspic1)));#normalize the image for uint8
endfunction

#normfull
function normpic = normfull(image);#handles neg value pics also
    vallo=min(min(image));
    if vallo<0
      image=image+vallo;
    endif
    valhi=max(max(image));
    normpic=uint8(image.*(250/valhi));
endfunction

#blknwht255
function bwpic255 = blknwht255(image); #convert image to black n white, any >0 value is max pixel value
  bwpic255=uint8(image.*255);
endfunction

#blknwht01
function bwpic01 = blknwht01(image); #convert image to black n white, any >0 value is max pixel value
  bwpic01=uint8(image.*255);
  bwpic01=uint8(bwpic01-254);
endfunction

#grad2 not uint8 , bi-pixel gradient
 function [dx2,dy2,dxy2] = grad2(pwt);#runs in less than a second vs. other 53sec me
    pwt=int16(pwt);
    x1=int16(shift(pwt,1,2));
    x2=int16(pwt);
    dx2=int16(x2-x1);
    y1=int16(shift(pwt,1,1));
    y2=int16(pwt);
    dy2=int16(y2-y1);
    xy1=int16(shift(x1,1,1));
    xy2=int16(shift(xy1,1,2));
    dxy2=int16(pwt-xy2);
    dx2(1,:)=0;dx2(2,:)=0;%set edges of win frame to zero for gradients
    dx2(size(pwt,1),:)=0;dx2((size(pwt,1))-1,:)=0;
    dx2(:,1)=0;dx2(:,2)=0;
    dx2(:,size(pwt,2))=0;dx2(:,(size(pwt,2))-1)=0;
    dy2(1,:)=0;dy2(2,:)=0;
    dy2(size(pwt,1),:)=0;dy2((size(pwt,1))-1,:)=0;
    dy2(:,1)=0;dy2(:,2)=0;
    dy2(:,size(pwt,2))=0; dy2(:,(size(pwt,2))-1)=0; 
    dxy2(1,:)=0;dxy2(2,:)=0;
    dxy2(size(pwt,1),:)=0;dxy2((size(pwt,1))-1,:)=0;
    dxy2(:,1)=0;dxy2(:,2)=0;
    dxy2(:,size(pwt,2))=0; dxy2(:,(size(pwt,2))-1)=0;
%    dx2=dx2+abs(min(min(dx2)));
%    dy2=dy2+abs(min(min(dy2)));
%    dxy2=dxy2+abs(min(min(dxy2)));#shift up out of negative values
endfunction

#grad3   Tri-pixel gradient, not uint8
function [dx3,dy3,dxy3] = grad3(pwt);#runs in less than a second vs. other 53sec me
    edge=0;
    pwt=int16(pwt);
    x1=shift(pwt,1,2);# 2 is the dimension upon to make the shift for x (clms)
    x2=shift(pwt,-1,2);
    dx3=x2-x1;
    y1=shift(pwt,1,1);
    y2=shift(pwt,-1,1);
    dy3=y2-y1;
    xy1=shift(x1,1,1);
    xy2=shift(y2,1,2);
    dxy3=xy2-xy1;
    dx3(1:(1+edge),:)=0;
    dx3(((size(pwt,1))-edge):size(pwt,1),:)=0;
    dx3(:,1:(1+edge))=0;
    dx3(:,((size(pwt,2))-edge):(size(pwt,2)))=0;
    dy3(1:(1+edge),:)=0;
    dy3(((size(pwt,1))-edge):size(pwt,1),:)=0;
    dy3(:,1:(1+edge))=0;
    dy3(:,((size(pwt,2))-edge):(size(pwt,2)))=0;
    dxy3(1:(1+edge),:)=0;
    dxy3(((size(pwt,1))-edge):size(pwt,1),:)=0;
    dxy3(:,1:(1+edge))=0;
    dxy3(:,((size(pwt,2))-edge):(size(pwt,2)))=0;
%    dx3=dx3+abs(min(min(dx3)));
%    dy3=dy3+abs(min(min(dy3)));
%    dxy3=dxy3+abs(min(min(dxy3)));
endfunction

#framediff2     Bi-frame differencing, not uint8
function [f] = framediff2(f1,f2);#f1 is most recent frame, f3 is two frames back in time
    f=int16(f1-f2);
endfunction

#framediff3     Tri-frame differencing, not uint8
function [f] = framediff3(f1,f2,f3);#f1 is most recent frame, f3 is two frames back in time
    s1=int16(f1-f2);
    s2=int16(f2-f3);
    f=int16(s2-s1);
endfunction

#Linear Contrast Thresholding across all divs
function threspic = lnrcontrastthres(imagethresdivs,imagethreswidth,pic);
    for m = 1:imagethresdivs;#cycle thru number of divisions of thresholding the image
      threslo=(1+(imagethreswidth*(m-1))); 
      threshi=imagethreswidth*m;
      p=threshilo(pic,threshi,threslo);
      threspic(:,:,m) = blknwht01(p);#threshold the most recent image to a new image blknwhite
    endfor
endfunction

#ccblobfinder   Connected Components Blob Finder
function [blob2 ccframe2] = ccblobfinder(img);#CONNECTED COMPONENTS BLOB DETECTOR
    #ccframe=worked image where pixels being replaced by markers of blobs as neg numbers, -1 is first blob, -2 is 2nd blob, etc..
    #blob=image of each recorded blob in 3rd dim (:,:,n) where n is nth blob count
    img=uint8(img);#Signed 8bit
    ccframe=int8(img); #img remains untouched through this algo, only used to get orig image values from
    [rf,cf]=size(ccframe);
    ccframe(:,1)=0;#set edges of ccframe to zeros because these are not used
    ccframe(:,cf)=0;
    ccframe(1,:)=0;
    ccframe(rf,:)=0;
    n=0;
    for r = 2:rf-1; #fix algo later to handle edges
      for c = 2:cf-1; #use neg numbrs for labeling pixels in ccframe, yet be able to keep orig values
        if ccframe(r,c)>0 #if the new pixel visited is >0, fresh new foreground pixel (it has not been visited before nor it's a zero val)
          n=n+1; #We are in a new blob, so increment the blob value index, 'n' is the name of the blob being marked
          blob(:,:,n)=uint8(zeros(rf,cf));#init this array with zeros, container for r,c of common n blobs
          blob(r,c,n)=img(r,c);#record the pixel for this blob, use n or 'size(blob,3)' value is the latest index of 3rd dim in blob
          ccframe(r,c)=-1*n;#we will mark each pixel with a neg value corresponding to the blob marking name, but a negative val of the blob name
        endif #do nothing if current pixel is zero
        if ccframe(r,c)<0 #if current pixel is not equal to zero then let's search it's neighbors, the above has already set the current blob label value
          ncurrent=abs(ccframe(r,c));#ncurrent is a positive value used for indexing the neg marker value
          if ccframe(r,c+1)!=0 #look east, check if already labeled, if so change it to current label
            if (ccframe(r,c+1)<0) && (ccframe(r,c+1)!=ccframe(r,c))#handles forks in image that spoofs cc algo 
              priorlabel=abs(ccframe(r,c+1));#cc algo is a raster scan, so we only need this check here for anti-spoofing
              blob(:,:,ncurrent)=blob(:,:,ncurrent)+blob(:,:,priorlabel);
              blob(:,:,priorlabel)=0;
            endif
            blob(r,c+1,ncurrent)=img(r,c+1);#place the pixel value into the blob array 
            ccframe(r,c+1)=ccframe(r,c);#apply current pixel 'label'(neg value) to this neighbor
          endif
          if ccframe(r+1,c+1)!=0 #look southeast, check if already labeled, if so change it to current label
            if (ccframe(r+1,c+1)<0) && (ccframe(r+1,c+1)!=ccframe(r,c))#handles forks in image that spoofs cc algo 
              priorlabel=abs(ccframe(r+1,c+1));#cc algo is a raster scan, so we only need this check here for anti-spoofing
              blob(:,:,ncurrent)=blob(:,:,ncurrent)+blob(:,:,priorlabel);
              blob(:,:,priorlabel)=0;
            endif
            blob(r+1,c+1,ncurrent)=img(r+1,c+1);
            ccframe(r+1,c+1)=ccframe(r,c);
          endif
          if ccframe(r+1,c)!=0 #look south, check if already labeled, if so change it to current label
            if (ccframe(r+1,c)<0) && (ccframe(r+1,c)!=ccframe(r,c))#handles forks in image that spoofs cc algo 
              priorlabel=abs(ccframe(r+1,c));#cc algo is a raster scan, so we only need this check here for anti-spoofing
              blob(:,:,ncurrent)=blob(:,:,ncurrent)+blob(:,:,priorlabel);
              blob(:,:,priorlabel)=0;
            endif
            blob(r+1,c,ncurrent)=img(r+1,c);
            ccframe(r+1,c)=ccframe(r,c);
          endif
          if ccframe(r+1,c-1)!=0 #look southwest, check if already labeled, if so change it to current label
            if (ccframe(r+1,c-1)<0) && (ccframe(r+1,c-1)!=ccframe(r,c))#handles forks in image that spoofs cc algo 
              priorlabel=abs(ccframe(r+1,c-1));#cc algo is a raster scan, so we only need this check here for anti-spoofing
              blob(:,:,ncurrent)=blob(:,:,ncurrent)+blob(:,:,priorlabel);
              blob(:,:,priorlabel)=0;
            endif
            blob(r+1,c-1,ncurrent)=img(r+1,c-1);
            ccframe(r+1,c-1)=ccframe(r,c);
          endif
        endif#break and do nothing if current pixel is zero
      endfor
    endfor
    x=0;#handles index lockup when blob(:,:,index) is reduced for removing the empty matrices in the below
    if n==0 #handles when there a NO blobs found
      ccframe2=[];#return value as empty array allowing to check using 'isempty' comparator in main program
      blob2=[]; #return value as empty array allowing to check using 'isempty' comparator in main program
      return;
    endif;
    for i=1:n #remove any blank blob dims and resort
      i=i-x; 
      if ~any(any(blob(:,:,i)))
        blob(:,:,i)=[];#destroy this 'i' blob frame
        x=x+1;
      endif
    endfor
    ccframe2=zeros(rf,cf);
    blob2=uint8((uint8(blob).*255)./255);#I can't recall why I did this, maybe something to do with emptying blob above and rebuilding ccframe2 below
    for i=1:size(blob,3)
      ccframe2=ccframe2+(i.*(blob2(:,:,i)));
    endfor
endfunction

#roiimage
function [picwin,ptrow,ptcol] = roiimage(pic,ptrow,ptcol,pw);#returned ptrow and ptcol are corrected locations limited to total picture bounds
    pw2=uint16(pw/2);#This subroutine makes a new smaller image centered at ptrow and ptcol of pw pixels width and height from 'pic'
    row1=ptrow-pw2;#first row, add code to handle edges of total pic
    row2=ptrow+pw2-1;#last row, -1 to keep size of PW exactly wide
    if row1<1 #check for window being out of bounds of picture
      row1=1; #fix the row index
      row2=pw; #fix 9subsequent indexing
      ptrow=row1+pw2;  #reset location of centerpoint wrt limited window edges
    endif
    if row2>size(pic,1)
      row2=size(pic,1);#set last row to end of pic rows
      row1=row2-pw;
      ptrow=row1+pw2; #ptrow position checked
    endif
    col1=ptcol-pw2;#first clm
    col2=ptcol+pw2-1;#last clm
    if col1<1
      col1=1;
      col2=pw;
      ptcol=col1+pw2;
    endif
    if col2>size(pic,2)
      col2=size(pic,2);
      col1=col2-pw;
      ptcol=col1+pw2; #ptcol position checked
    endif
    picwin=pic(row1:row2,col1:col2); #set subwindow (ROI/region of interest) from pic
endfunction

function dspic = windowdrawsquare(image,i,j,w);
  #image is the picture
  #i is row of center point
  #j is column of center point
  dspic=image;
    #window area box
  dspic(round(i-w/5):round(i+w/5),round(j-w/2))=1; #left vert
  dspic(round(i-w/5):round(i+w/5),round(j+w/2))=1; #right vert
  dspic(round(i+w/2),round(j-w/5):round(j+w/5))=1; #top horz
  dspic(round(i-w/2),round(j-w/5):round(j+w/5))=1; #bot horz
endfunction

#pointdrawsquare  positive pixels are set herein as 'value'
function dspic = pointdrawsquare(image,i,j,pw,value);#i is ptrow (Y) and j is ptcol (X)pw is total width and value is set pixel value
  #image is the picture
  #i is row of center point
  #j is column of center point
  dspic=image;
  #draw a small box whatever pixels wide
  pw=uint16(pw);
  x=uint16(pw/2);#4;
  i=uint16(i);
  j=uint16(j);
  n=uint16(x);
  e=uint16(x);
  s=uint16(x);
  w=uint16(x);#cross hair arm lengths, north easht south and west arm lengths set
  #******
  w1=uint16(i-n); w2=uint16(i+s); w3=uint16(j-w); #use uint16 to remove negative values
  if w1==0 #can't use zero index in image
    w1=1;
  endif
  if w2==0
    w2=1;
  endif
  if w3==0
    w3=1;
  endif
  dspic((w1):(w2),(w3))=uint8(value); #left/west vert line
  #******
  e1=uint16(i-n); e2=uint16(i+s); e3=uint16(j+e);
  if e1==0 #can't use zero index in image
    e1=1;
  endif
  if e2==0
    e2=1;
  endif
  if e3==0
    e3=1;
  endif
  dspic(e1:e2,e3)=uint8(value); #right/east vert
  #******
  n1=uint16(j-e); n2=uint16(j+w); n3=uint16(i+s);
  if n1==0 #can't use zero index in image
    n1=1;
  endif
  if n2==0
    n2=1;
  endif
  if n3==0
    n3=1;
  endif
  dspic(n3,n1:n2)=uint8(value); #top/north horz
  #******
  s1=uint16(j-e); s2=uint16(j+w); s3=uint16(i-n);
  if s1==0 #can't use zero index in image
    s1=1;
  endif
  if s2==0
    s2=1;
  endif
  if s3==0
    s3=1;
  endif
  dspic(s3,s1:s2)=uint8(value); #bottom/south horz
  #*******
  dspic=dspic(1:size(image,1),1:size(image,2)); #removes any lines generated outside the frame
endfunction

#pointdrawplus  in size s and pixel value of 'value'
function dspic = pointdrawplus(image,i,j,pw,value); #i is ptrow (Y) and j is ptcol (X)pw is total width and value is set pixel value
  #image is the picture
  #i is row of center point
  #j is column of center point
  dspic=image;
  #draw a plus sign x pixels long
  pw=uint16(pw);
  x=uint16(pw/2);#4;
  i=uint16(i);
  j=uint16(j);
  n=uint16(x);
  e=uint16(x);
  s=uint16(x);
  w=uint16(x);#cross hair arm lengths, north easht south and west arm lengths set
  if i<(1+n) #i is ptrow (Y) check if close to image axis
    n=uint16(i-1); #if so, then arm of cross hair is shortened, must be zero
  endif
  if i>(size(image,1)-(1+s))#check if close to image extremeties
    s=(i+1)-(uint16(size(image,1)));#if so, then box center is pushed away from edge but keeps the intended size
  endif
  if j<(1+w)
    w=uint16(j-1);
  endif
  if j>(size(image,2)-(1+e))
    e=(j+1)-(uint16(size(image,1)));
  endif
  dspic((i-n):(i+s),(j))=uint8(value);#uint8(value); #center vert
  dspic((i),(j-e):(j+w))=uint8(value);#uint8(value); #center horz
endfunction


#pointdrawplus  in size s and pixel value of 'value'
function dspic = pointfullplus(image,i,j,value); #i is ptrow (Y) and j is ptcol (X)pw is total width and value is set pixel value
  #image is the picture
  #i is row of center point
  #j is column of center point
  dspic=image;
  #draw a plus sign x pixels long
  i=uint16(i);
  j=uint16(j);
  if i==0
    i=1;
  endif
  if j==0
    j=1;
  endif
  dspic(:,(j))=uint8(value); #center vert
  dspic((i),:)=uint8(value); #center horz
  dspic=dspic(1:size(image,1),1:size(image,2)); #removes any lines generated outside the frame
endfunction

function [startrow stoprow startcol stopcol] = findimagedges(img);
    rowsummed=sum(img,1);#X vals results a single row vector, sum of rows leaving X vals summed as colstyle
    colsummed=sum(img,2);#Y vals results a single column vector
    for d = [1:size(rowsummed,2)]#x direction , columns
      if rowsummed(d) > 0 
        startcol=d;
        break#exit from for loop, or this loop will keep finding nonzero values until the last one, this requres starts to be stop
      endif   #using the break is faster since it runs less times instead of all the way thru pixel rows
    endfor
    for d = [size(rowsummed,2):-1:1]#x direction , columns
      if rowsummed(d) > 0 
        stopcol=d;
        break
      endif   
    endfor
    for d = [1:size(colsummed,1)]#y direction , rows
      if colsummed(d) > 0 
        startrow=d;
        break
      endif   
    endfor
    for d = [size(colsummed,1):-1:1]#y direction , rows
      if colsummed(d) > 0 
        stoprow=d;
        break
      endif   
    endfor
endfunction
