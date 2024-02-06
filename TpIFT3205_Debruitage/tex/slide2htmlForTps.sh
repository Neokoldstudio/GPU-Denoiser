
#################################################
# Usage: slide2html slide.dvi slide.html slide  #
#                                               #
# creates slide.html                            #
# creates a directory slide.html_slides         # 
# put all slides inside that directory          # 
#                                               #
#################################################

echo ""
echo "--- slides for $1 to $2 ---"
echo ""

slides=${3}_slides
echo "I have $slides"

if test -d $slides ; then
	echo "$slides already exist!"
        echo ""
	rm -r $slides
else
	echo "$slides does not exist!"
        echo ""
fi

mkdir $slides

out=${slides}/slide%03d.png
iout=${slides}/icon%03d.png

##############################################################
# make a bunch of files with full aliasing and downsampling  #
##############################################################

dvips -O 0in,-0.5in -t letter -o "!gs -dNOPAUSE -r144 -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -g1200x1750 -sDEVICE=ppmraw -sOutputFile=\"|pnmscale 0.7 |pnmtopng >$out\" - -c quit" $1


####################
# make small icons #
####################

dvips -O 0in,-0.5in -t letter -o "!gs -dNOPAUSE -r72 -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -g600x875 -sDEVICE=ppmraw -sOutputFile=\"|pnmscale 0.45 |pnmtopng >$iout\" - -c quit" $1


############################
# how many slides, really? #
############################

nbp=`/bin/ls $slides |grep icon | wc | awk '{print $1;}'`

echo ""
echo "We got $nbp slides"
echo ""

##########################
# make a crude html page #
##########################

echo "<HTML><HEAD></HEAD></HTML>" >$2
echo "<BODY LINK="#000FFF" BACKGROUND="Images_IFT3205/SP_Slides_BckGrnd.gif">" >>$2

echo "<CENTER>" >>$2
echo "<TABLE>" >>$2

echo "<TR bgcolor=#7777cc><TH colspan=5>$3</TH></TR>" >>$2

i=1
nbpp=`expr $nbp + 1`
while [ $i != $nbpp ]; do
    j=`expr $i - 1`
    j=`expr $j % 5`
    k=`printf %03d $i`

    if [ $j = 0 ]; then
	echo "<TR>" >>$2
    fi
    echo "<TD><A HREF=\"$slides/slide$k.html\"><IMG SRC=\"$slides/icon$k.png\" BORDER=0></A></TD>" >>$2
    if [ $j = 4 ]; then
	echo "</TR>" >>$2
    fi

    ####################################
    # add a little file for each image #
    ####################################
    kp=`expr $k - 1`
    kn=`expr $k + 1`
    kp=`printf %03d $kp`
    kn=`printf %03d $kn`
    f="$slides/slide$k.html"
    echo "<HTML><HEAD></HEAD>" >>$f
    echo "<BODY LINK="#000FFF" BACKGROUND="../Images_IFT3205/SP_Slides_BckGrnd.gif">" >>$f
    echo "<CENTER>" >>$f
    echo "<TABLE CELLSPACING=0 CELLPADDING=4 BORDER=0>" >>$f
    echo "<TR bgcolor=#7777cc>" >>$f

    if [ $i != 1 ]; then
    echo "<TD align=left><A HREF=\"slide$kp.html\"><IMG SRC=\"flecheleftgrey.gif\" border=0></A>" >>$f
    else
    echo "<TD align=left><IMG SRC=\"flechefill.gif\" border=0>" >>$f
    fi

    echo "<A HREF=\"../$2\"><IMG SRC=\"flecheupgrey.gif\" border=0></A>" >>$f

    if [ $i != $nbp ]; then
    echo "<A HREF=\"slide$kn.html\"><IMG SRC=\"flecherightgrey.gif\" border=0></A>" >>$f
    else
    echo "<IMG SRC=\"flechefill.gif\" border=0>" >>$f
    fi
    g=`expr $i - 1`
    echo "</TD><TD align=right><FONT SIZE=+2>$g</FONT></TD></TR>" >>$f

    echo "<TD colspan=2><IMG SRC=\"slide$k.png\" BORDER=0></TD>" >>$f
    echo "</TR>" >>$f
    echo "</TABLE>" >>$f
    echo "<BR>" >>$f
    echo "<A HREF="http://www.iro.umontreal.ca/~mignotte/ift3205">IFT3205</A>" >>$f
    echo "</CENTER>" >>$f
    echo "</BODY></HTML>" >>$f

    i=`expr $i + 1`
done

echo "</TABLE>" >>$2
echo "<BR>" >>$2
echo "<A HREF="http://www.iro.umontreal.ca/~mignotte/ift3205">IFT3205</A>" >>$2
echo "<BR>" >>$2
echo "<BR>" >>$2
echo "<BR>" >>$2
echo "<BR>" >>$2
echo "</CENTER>" >>$2
echo "</BODY></HTML>" >>$2

###############
# copy images #
###############

cp /u/mignotte/HTML/image_slide/flecheupgrey.gif $slides
cp /u/mignotte/HTML/image_slide/flecheleftgrey.gif $slides
cp /u/mignotte/HTML/image_slide/flecherightgrey.gif $slides
cp /u/mignotte/HTML/image_slide/flechefill.gif $slides


