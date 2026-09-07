#include <stdio.h>

#include "../deepseek_v4_model.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s SNAPSHOT\n", argv[0]);
        return 2;
    }
    dsv4_store store;
    if (!dsv4_store_init(&store, argv[1])) {
        fprintf(stderr, "store: %s\n", store.error);
        dsv4_store_close(&store);
        return 1;
    }
    dsv4_model_contract contract;
    if (!dsv4_model_bind_contract(&contract, &store)) {
        fprintf(stderr, "contract: %s\n", contract.error);
        dsv4_store_close(&store);
        return 1;
    }
    printf("DeepSeek-V4 contract: tensors=%d layers=%d dspark=%d\n",
           store.records, contract.base_layers, contract.dspark_layers);
    dsv4_store_close(&store);
    return 0;
}
