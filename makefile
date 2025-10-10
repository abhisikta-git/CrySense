# file := $(FILE)
FILE := $(file)
FILE := $(f)

.PHONY: help train preprocess predict convert visualize vis v clean

# Default target
help:
	@echo "CrySense - Baby Cry Classifier"
	@echo ""
	@echo "Available commands:"
	@echo "  make train                    - Train the model"
	@echo "  make preprocess FILE=audio.wav- Clean the raw/recorded audio"
	@echo "  make predict FILE=audio.wav   - Predict audio classification"
	@echo "  make visualize FILE=audio.wav - Visualize audio (alias: vis, v)"
	@echo "  make clean                    - Remove 'outputs' and 'models' folder"
	@echo ""
	@echo "Note: FILE, file, and f work as parameter names"
	@echo ""
	@echo "example:" 
	@echo "make preprocess FILE=audio.wav"
	@echo "make preprocess file=audio.wav"
	@echo "make preprocess f=audio.wav"
	@echo ""
	@echo "all 3 works the same"
	@echo ""

train:
	python ./src/train.py

predict:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make predict FILE=path/to/audio.wav"; \
	else \
		python ./src/predict.py $(FILE); \
	fi

convert:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make convert FILE=path/to/audio.wav"; \
	else \
		python ./src/convert_to_wav.py $(FILE); \
	fi

preprocess:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make preprocess FILE=path/to/audio.wav"; \
	else \
		python ./src/preprocess.py $(FILE); \
	fi

visualize:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make visualize FILE=path/to/audio.wav"; \
	else \
		python ./src/visualize.py $(FILE); \
	fi

# aliases
vis: visualize
v: visualize

clean:
	trash -f \
	./models/baby_cry_model.keras \
	./models/label_encoder.pkl \
	./outputs/visualizations