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
	struct Val *val;
	struct Tnode *next;
};

void topo_prepend(struct Tnode **topo, struct Val **val) {
	struct Tnode *n_tnode = (struct Tnode *)malloc((sizeof(struct Tnode)));
	n_tnode->val  = *val;
	n_tnode->next = *topo;
	*topo = n_tnode;
}

void topo_build(struct Val **val, struct Tnode **topo, struct Val **visited, size_t *visited_count) {
	for (size_t i = 0; i < *visited_count; i++) {
		if (visited[i] == *val) {
			return;
		}
	}

	visited[*visited_count] = *val;
	(*visited_count)++;

	for (size_t i = 0; i < 2; i++) {
		topo_build((*val)->children[i], topo, visited, visited_count);
	}

	topo_prepend(topo, val);
}

void topo_free(struct Tnode *topo) {
	struct Tnode *current = topo;
	while (current != NULL) {
		struct Tnode *next = current->next;
		free(current);
		current = next;
		}
}

void val_backward(struct Val **val) {
	struct Tnode *topo = NULL;
	struct Val **visited = (struct Val **)malloc(sizeof(struct Val *) * 100); /* assuming max 100 vals */
	size_t visited_count = 0;

	(*val)->grad = 1.0;
	topo_build(val, &topo, visited, &visited_count);

	for (size_t i = visited_count; i > 0; i--) {
		topo[i].val->backward(topo[i].val->children[0], topo[i].val->children[1], &topo[i].val);
	}

	free(visited);
	topo_free(topo);
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

int main(void) {
	struct Val *val1 = val_init(2.5);
	struct Val *val2 = val_init(3.5);
	struct Val *val3 = val_add(&val1, &val2);

	val_backward(&val3);

	val_print(*val1);
	val_print(*val2);
	val_print(*val3);

	free(val1);
	free(val2);
	free(val3);

	return 0;
}