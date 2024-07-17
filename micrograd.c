#include <stdio.h>
#include <stdlib.h>

struct Val {
	float data;
	float grad;
	void (*backward)(struct Val **, struct Val **, struct Val **); 
	struct Val **children[2];
	char op;
};

struct Tnode {
	Val *val;
	struct Tnode *next;
};

void topo_append(Tnode **topo, Val *val) {
}

void topo_build() {
}

void topo_free() {
}

// mul backward
void add_backward(struct Val **val1, struct Val **val2, struct Val **out) {
	(*val1)->grad += 1.0 * (*out)->grad;
	(*val2)->grad += 1.0 * (*out)->grad;
}

struct Val *val_init(float n_data) {
	struct Val *val = (struct Val *)malloc(sizeof(struct Val));
	val->data = n_data;
	val->grad = 0.0;
	val->backward = NULL;
	val->op = '\0';

	return val;
}

void val_print(struct Val val) {
	printf("val -> data: %.4f, grad: %.4f, op: '%c'\n", val.data, val.grad, val.op);
}

void val_free(struct Val *val) {
	/* dont't forget to free children */
	free(val);
}

struct Val *val_add(struct Val **val1, struct Val **val2) {
	struct Val *out = val_init((*val1)->data + (*val2)->data);
	out->op = '+';
	out->backward = &add_backward;

	out->children[0] = val1;
	out->children[1] = val2;

	return out;
}
// struct Val *val_mul

void val_backward(struct Val **val) {
	// topo = {}

	// for node in topo
	// 		node.backward
}

int main(void) {
	struct Val *val1 = val_init(2.5);
	struct Val *val2 = val_init(3.5);
	struct Val *val3 = val_add(&val1, &val2);
	val3->grad = 1.0;
	val3->backward(val3->children[0], val3->children[1], &val3);

	val_print(*val1);
	val_print(*val2);
	val_print(*val3);

	free(val1);
	free(val2);
	free(val3);

	return 0;
}