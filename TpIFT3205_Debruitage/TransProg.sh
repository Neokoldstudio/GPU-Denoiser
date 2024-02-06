#! /bin/sh
# TransProg.sh script
#----------------------


#--------------
#--WEB ADDRESS
#--------------
AddressWeb="/u/mignotte/HTML/IFT3205/ProgTpIFT3205/"

#--------------
#--NameTp
#--------------

NameTp=`pwd | cut -d "_" -f 3`
NameDirTp=ProgTpIFT3205_${NameTp}
PathDirTp=${AddressWeb}$NameDirTp

echo Repertoire ${NameTp}

#-----------------------
#--Creation Repertoire
#-----------------------

if [ ! \( -d $PathDirTp \) ]
then
echo Je dois creer $PathDirTp
mkdir $PathDirTp
chmod 755 $PathDirTp
else
echo Le repertoire $PathDirTp existe deja
chmod 755 $PathDirTp
fi

#-----------------------------
#--Transfert Prog/Bareme/Mail
#-----------------------------

echo "->" Transfert Progs
cp prog/* ${PathDirTp}/.

echo "->" Transfert Bareme
cp tex/Bareme_IFT3205_${NameTp}.dat ${PathDirTp}/Bareme_IFT3205_${NameTp}.dat

echo "->" Transfert EMail
cp tex/Mail_IFT3205_${NameTp}.dat ${PathDirTp}/EMail_IFT3205_${NameTp}.dat


#-----------------------------
#--Chmod 
#-----------------------------

chmod 755 ${PathDirTp}/*

#-------------------------
#--End
#-------------------------
echo "-> Fini ..."
