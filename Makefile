train_agnews:
	python train.py \
	    --use-dvat \
	    --n-labeled ${N_LABELED} \
	    --tsa 'lin_schedule' \
	    --confidence 0.0 \
	    --seed ${SEED} \
	    --normalize-grad 'inf' \
	    --learning-rate ${LR} \
	    --swap-ratio 0.25 \
	    --topk 10 \
	    --data-path data/AG_NEWS/


train_yahoo:
	python train.py \
	    --use-dvat \
	    --n-labeled ${N_LABELED} \
	    --tsa 'lin_schedule' \
	    --confidence 0.0 \
	    --seed ${SEED} \
	    --normalize-grad 'inf' \
	    --learning-rate ${LR} \
	    --swap-ratio 0.25 \
	    --topk 10 \
	    --data-path data/YAHOO/


train_dbpedia:
	python train.py \
	    --use-dvat \
	    --n-labeled ${N_LABELED} \
	    --tsa 'lin_schedule' \
	    --confidence 0.0 \
	    --seed ${SEED} \
	    --normalize-grad 'inf' \
	    --learning-rate ${LR} \
	    --swap-ratio 0.25 \
	    --topk 10 \
	    --data-path data/DBpedia/