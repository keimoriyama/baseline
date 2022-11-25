train:
	rm -rf logs/logs.txt
	rm -rf logs/stdout.txt
	qsub -g gcc50441 train.sh