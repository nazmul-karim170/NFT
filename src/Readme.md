To train a backdoor model with "blend" attack with poison ratio of "10%"-

	python train_backdoor_cifar.py --poison-type blend --poison-rate 0.10 --output-dir save/blend/ --gpuid 0 

To train a benign model-

    python train_backdoor_cifar.py --poison-type benign --output-dir save/benign/ --gpuid 0 


To remove backdoor-
	
	python Remove_Backdoor.py --poison-type blend --checkpoint save/blend/ --gpuid 0 


