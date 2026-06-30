ts=$( date "+%Y-%m-%d" )
fl=conference/*.txt
IFS= read -r -p "Enter a keyword: " input
outf=$(echo $input | sed 's/\ /_/')
  
echo keyword on $input >keyword_$outf'_'$ts.txt
#for keyword in $input
#do
keyword=$input
for ff in $fl 
do
  ss=$(cat $ff | grep -i "$keyword" | wc -c)
  if [ $ss -gt 0 ];
  then
    echo '' >> keyword_$outf'_'$ts.txt
    echo '####' $ff >>keyword_$outf'_'$ts.txt
   cat $ff | grep -i "$keyword" >>keyword_$outf'_'$ts.txt
  fi
done
#done
