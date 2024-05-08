import logging
from args import parse_train_opt
from POPDG import POPDG
import traceback

logging.basicConfig(filename='training.log',level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(opt):
    """Train the model based on the provided options."""
    logging.info("Initializing the model...")
    try:
        model = POPDG(opt.feature_type)
    except Exception as e:
        logging.error("Failed to initialize the EDGE model: %s", e)
        logging.error("Exception details:\n%s", traceback.format_exc())
        return  

    logging.info("Starting the training loop...")
    try:
        model.train_loop(opt)
    except Exception as e:
        logging.error("An error occurred during training: %s", e)
        logging.error("Exception details:\n%s", traceback.format_exc())
        return  

def main():
    """Parse options and run the training process."""
    logging.info("Parsing training options...")
    opt = parse_train_opt()
    
    if opt is not None:
        train(opt)
    else:
        logging.error("No valid training options provided.")

if __name__ == "__main__":
    main()
