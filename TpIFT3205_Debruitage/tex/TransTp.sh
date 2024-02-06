#! /bin/sh
# Transfile.sh script
#----------------------

#--------------
#--NameFile
#--------------
NameFileDef="TpIFT3205_Debruitage"
NameFile=${1:-${NameFileDef}}
echo $NameFile

#--------------
#--WEB ADDRESS
#--------------
AddressWeb="/u/mignotte/HTML/IFT3205/"

#-------------------------
#--PDF/PDF2x Conversion
#-------------------------
echo "-> Postscript file"
dvips -t letter ${NameFile}.dvi -o ${NameFile}.ps

echo "-> Conversion PDF"
ps2pdf ${NameFile}.ps ${NameFile}.pdf

#-------------------------
#--PDF/PDF2x Transfert
#-------------------------
echo "-> Transfert PDF"
mv ${NameFile}.pdf    ${AddressWeb}    
chmod 755 ${AddressWeb}*

#-------------------------
#--HTML Conversion
#-------------------------
echo "-> HTML Conversion"
slide2htmlForTps.sh ${NameFile}.dvi  ${NameFile}.html ${NameFile}

#-------------------------
#--HTML Transfert
#-------------------------
echo "-> Transfert Main Page Web"
mv ${NameFile}.html ${AddressWeb}  
chmod 755   ${AddressWeb}*

echo "-> Transfert Fichiers Page Web"

if [ -d ${AddressWeb}${NameFile}_slides ]
then
echo Repertoire Existe Copiage
cp ${NameFile}_slides/* ${AddressWeb}${NameFile}_slides/. 
echo Menage
echo On efface ${NameFile}_slides
rm -r ${NameFile}_slides
else
echo repertoire existe pas on move
mv ${NameFile}_slides ${AddressWeb} 
fi

echo "-> Chmod Page Weib"
chmod 755 ${AddressWeb}${NameFile}_slides 
chmod 755 ${AddressWeb}${NameFile}_slides/*

#-------------------------
#--CleanUp
#-------------------------
echo "-> CleanUp Ps"
rm ${NameFile}.ps

#-------------------------
#--End
#-------------------------
echo "-> Fini ..."
