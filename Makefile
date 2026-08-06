.PHONY: all qwen test test-c test-python test-tokenizer test-docs test-source-package test-web check clean install
all qwen test test-c test-python test-tokenizer test-docs test-source-package test-web check clean install:
	$(MAKE) -C c $@
