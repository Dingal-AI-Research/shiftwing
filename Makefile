.PHONY: all qwen test test-c check clean install
all qwen test test-c check clean install:
	$(MAKE) -C c $@
