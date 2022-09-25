import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from paead_info import *
from paead_models import LitBART, LitMSBART, LitMSBARTwithSlot2Intent, LitBERTTag


def train(model, summary_data, disable_training=False):
	if isinstance(model, LitBART):
		logger_dir = LB_CHECKPOINT_DIR
		epochs = LB_EPOCHS
		reload_epochs = 0
	elif isinstance(model, LitMSBARTwithSlot2Intent):
		logger_dir = LMSBwS2I_CHECKPOINT_DIR
		epochs = LMSBwS2I_EPOCHS
		reload_epochs = LMSBwS2I_HATE_PRETRAIN
	elif isinstance(model, LitMSBART):
		logger_dir = LMSB_CHECKPOINT_DIR
		epochs = LMSB_EPOCHS
		reload_epochs = LMSB_HATE_PRETRAIN
	elif isinstance(model, LitBERTTag):
		logger_dir = BERTTAG_CHECKPOINT_DIR
		epochs = BERTTAG_EPOCHS
		reload_epochs = BERTTAG_HATE_PRETRAIN
	else:
		assert False, 'Logger dir has not been set'
	tb_logger = pl_loggers.TensorBoardLogger(logger_dir)
	checkpoint = ModelCheckpoint(filename='{epoch}-{step}-{val_prod_f1:.4f}', monitor='val_prod_f1', mode='max', save_top_k=1, every_n_epochs=1, save_on_train_epoch_end=True, auto_insert_metric_name=True)
	trainer = pl.Trainer(gpus = 1,
					logger = tb_logger,
	                max_epochs = epochs,
	                min_epochs = 1,
	                auto_lr_find = False,
	                checkpoint_callback = True,
	                callbacks = checkpoint,
	                reload_dataloaders_every_n_epochs = reload_epochs,
	                progress_bar_refresh_rate = 10  # , deterministic=True
                    )
	# Fit the instantiated model to the data
	if not disable_training:
		trainer.fit(model, summary_data)
	return trainer
