//see: http://karpathy.github.io/neuralnets/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>

typedef struct {
	float value;
	float grad;
} Unit;

typedef struct multiplyGate{
	Unit *u0;
	Unit *u1;
	Unit utop;
	Unit (*(*forward)(struct multiplyGate *this, Unit *u0, Unit *u1));
	void (*backward)(struct multiplyGate *this);
} multiplyGate;

Unit* forward_multiplyGate(multiplyGate *this, Unit *u0, Unit *u1) {
	this->u0 = u0;
	this->u1 = u1;
	this->utop.value = u0->value * u1->value;
	this->utop.grad = 0;
	return &this->utop;
}

void backward_multiplyGate(multiplyGate *this) {
	this->u0->grad += this->u1->value * this->utop.grad;
	this->u1->grad += this->u0->value * this->utop.grad;
}

multiplyGate* new_multiplyGate() {
	multiplyGate *mulg0 = malloc(sizeof(multiplyGate));
	mulg0->forward = forward_multiplyGate;
	mulg0->backward = backward_multiplyGate;
	return mulg0;
}

typedef struct addGate {
	Unit *u0;
	Unit *u1;
	Unit utop;
	Unit (*(*forward)(struct addGate *this, Unit *u0, Unit *u1));
	void (*backward)(struct addGate *this);
} addGate;

Unit* forward_addGate(addGate *this, Unit *u0, Unit *u1) {
	this->u0 = u0;
	this->u1 = u1;
	this->utop.value = u0->value + u1->value;
	this->utop.grad = 0;
	return &this->utop;
}

void backward_addGate(addGate *this) {
	this->u0->grad += 1 * this->utop.grad;
	this->u1->grad += 1 * this->utop.grad;
}

addGate* new_addGate() {
	addGate *addg0 = malloc(sizeof(addGate));
	addg0->forward = forward_addGate;
	addg0->backward = backward_addGate;
	return addg0;
}

typedef struct ReLuGate {
	Unit *u0;
	Unit utop;
	float (* ReLu)(float x);
	Unit (*(*forward)(struct ReLuGate *this, Unit *u0));
	void (*backward)(struct ReLuGate *this);
} ReLuGate;

float ReLu(float x) {
	return x > 1 ? 1 : x > 0 ? x : 0;
}

Unit* forward_ReLuGate(ReLuGate *this, Unit *u0) {
	this->u0 = u0;
	this->utop.value = this->ReLu(u0->value);
	this->utop.grad = 0;
	return &this->utop;
}

void backward_ReLuGate(ReLuGate *this) {
	float s = this->ReLu(this->u0->value);
	//this->u0->grad += (s > 1 ? 0 : (s > 0 ? 1 : 0)) * this->utop.grad;
	this->u0->grad += (s > 0 ? 1 : 0) * this->utop.grad;
}

ReLuGate* new_ReLuGate() {
	ReLuGate *reluGate = malloc(sizeof(ReLuGate));
	reluGate->ReLu = ReLu;
	reluGate->forward = forward_ReLuGate;
	reluGate->backward = backward_ReLuGate;
	return reluGate;
}

typedef struct sigmoidGate {
	Unit *u0;
	Unit utop;
	float (* sigmoid)(float x);
	Unit (*(*forward)(struct sigmoidGate *this, Unit *u0));
	void (*backward)(struct sigmoidGate *this);
} sigmoidGate;

float sigmoid(float x) {
	return 1.0 / (1 + exp(-x));
}

Unit* forward_sigmoidGate(sigmoidGate *this, Unit *u0) {
	this->u0 = u0;
	this->utop.value = this->sigmoid(u0->value);
	this->utop.grad = 0;
	return &this->utop;
}

void backward_sigmoidGate(sigmoidGate *this) {
	float s = this->sigmoid(this->u0->value);
	this->u0->grad += (s * (1 - s)) * this->utop.grad;
}

sigmoidGate* new_sigmoidGate() {
	sigmoidGate *sg = malloc(sizeof(sigmoidGate));
	sg->sigmoid = sigmoid;
	sg->forward = forward_sigmoidGate;
	sg->backward = backward_sigmoidGate;
	return sg;
}

typedef struct Circuit {
	multiplyGate *mulg0;
	multiplyGate *mulg1;
	addGate *addg0;
	addGate *addg1;
	ReLuGate *sGate;

	Unit *ax;
	Unit *by;
	Unit *axpby;
	Unit *axpbypc;
	Unit *sValue;

	Unit (*(*forward)(struct Circuit *this, Unit *x, Unit *y, Unit *a, Unit *b, Unit *c));
	void (*backward)(struct Circuit *this, float gradient_top);
} Circuit;

void init_Circuit(struct Circuit *this) {
	this->mulg0 = new_multiplyGate();
	this->mulg1 = new_multiplyGate();
	this->addg0 = new_addGate();
	this->addg1 = new_addGate();
	this->sGate = new_ReLuGate();
}

Unit* forward_Circuit(Circuit *this, Unit *x, Unit *y, Unit *a, Unit *b, Unit *c) {
	this->ax = this->mulg0->forward(this->mulg0, a, x); // a*x
	this->by = this->mulg1->forward(this->mulg1, b, y); // b*y
	this->axpby = this->addg0->forward(this->addg0, this->ax, this->by); // a*x + b*y
	this->axpbypc = this->addg1->forward(this->addg1, this->axpby, c); // a*x + b*y + c
	this->sValue = this->sGate->forward(this->sGate, this->axpbypc);
	return this->sValue;
}

void backward_Circuit(struct Circuit *this, float gradient_top) {
	this->sValue->grad = gradient_top;
	this->sGate->backward(this->sGate);
	this->addg1->backward(this->addg1); // sets gradient in axpby and c
	this->addg0->backward(this->addg0); // sets gradient in ax and by
	this->mulg1->backward(this->mulg1); // sets gradient in b and y
	this->mulg0->backward(this->mulg0); // sets gradient in a and x
}

Circuit* new_Circuit() {
	Circuit *circuit = malloc(sizeof(Circuit));
	circuit->forward = forward_Circuit;
	circuit->backward = backward_Circuit;
	init_Circuit(circuit);
	return circuit;
}

void TestCircuit2() {
	Circuit *circuit = new_Circuit();

	Unit x = { .value = 0.1, 0 };
	Unit y = { .value = 0.2, 0 };
	Unit a = { .value = 0.3, 0 };
	Unit b = { .value = 0.4, 0 };
	Unit c = { .value = 0.5, 0 };
	
	Unit* unit_out = circuit->forward(circuit, &x, &y, &a, &b, &c);

	// ax + by + c = 0.61
	assert((int)(unit_out->value*100) == 61);

	free(circuit);

	printf("TestCircuit2 [passed]\n");
}

void TestCircuit_Sigmoid() {
	Circuit *circuit = new_Circuit();

	Unit a = { .value = 1.0, 0 };
	Unit b = { .value = 2.0, 0 };
	Unit c = { .value = -3.0, 0 };
	Unit x = { .value = -1.0, 0 };
	Unit y = { .value = 3.0, 0 };
	
	Unit* unit_out = circuit->forward(circuit, &x, &y, &a, &b, &c);
	printf("s: %f\n", unit_out->value);

	printf("s: %f %f\n", unit_out->value, unit_out->grad);
	circuit->backward(circuit, 1.0);
	printf("s: %f %f\n", unit_out->value, unit_out->grad);

	float step_size = 0.01;
	a.value += step_size * a.grad;
	b.value += step_size * b.grad;
	c.value += step_size * c.grad;
	x.value += step_size * x.grad;
	y.value += step_size * y.grad;
	unit_out = circuit->forward(circuit, &x, &y, &a, &b, &c);
	printf("s: %f\n", unit_out->value);

	assert((int)(unit_out->value*1000000) == 882550);

	free(circuit);

	printf("TestCircuit [passed]\n");
}

void TestCircuit() {
	printf("TestCircuit: ReLu\n");

	Circuit *circuit = new_Circuit();

	Unit a = { .value = 0.1, 0 };
	Unit b = { .value = 0.2, 0 };
	Unit c = { .value = 0.3, 0 };
	Unit x = { .value = 0.1, 0 };
	Unit y = { .value = 0.3, 0 };
	
	Unit* unit_out = circuit->forward(circuit, &x, &y, &a, &b, &c);

	printf("s: %f %f\n", unit_out->value, unit_out->grad);
	circuit->backward(circuit, 1.0);
	printf("s: %f %f\n", unit_out->value, unit_out->grad);

	float step_size = 0.01;
	a.value += step_size * a.grad;
	b.value += step_size * b.grad;
	c.value += step_size * c.grad;
	x.value += step_size * x.grad;
	y.value += step_size * y.grad;
	unit_out = circuit->forward(circuit, &x, &y, &a, &b, &c);
	printf("s: %f\n", unit_out->value);

	assert((int)(unit_out->value*1000000) == 381507);

	free(circuit);

	printf("TestCircuit [passed]\n");
}

typedef struct SVM {
	Unit a1;
	Unit b1;
	Unit c1;

	Unit a2;
	Unit b2;
	Unit c2;
	
	Unit a3;
	Unit b3;
	Unit c3;
	
	Unit *unit_c1out;
	Unit *unit_c2out;
	Unit unit_out;
	
	Circuit *circuit1;
	Circuit *circuit2;
	Circuit *circuit3;
	
	Unit (*(*forward)(struct SVM *this, Unit *x, Unit *y));
	void (*backward)(struct SVM *this, int label);
	void (*parameterUpdate)(struct SVM *this);
	void (*learnFrom)(struct SVM *this, Unit *x, Unit *y, int label);
} SVM;

Unit* forward_SVM(SVM *this, Unit *x, Unit *y) {
	this->unit_c1out = this->circuit1->forward(this->circuit1, x, y, &this->a1, &this->b1, &this->c1);
	this->unit_c2out = this->circuit2->forward(this->circuit2, x, y, &this->a2, &this->b2, &this->c2);
	this->unit_out = *this->circuit3->forward(this->circuit3, this->unit_c1out, this->unit_c2out, &this->a3, &this->b3, &this->c3);
	return &this->unit_out;
}

void backward_SVM(SVM *this, int label) {
	this->a1.grad = 0;
	this->b1.grad = 0;
	this->c1.grad = 0;

	this->a2.grad = 0;
	this->b2.grad = 0;
	this->c2.grad = 0;
	
	this->a3.grad = 0;
	this->b3.grad = 0;
	this->c3.grad = 0;

	int pull = 0;

	if(label == 1 && this->unit_out.value < 0.7) { 
	  pull = 1; // the score was too low: pull up
	}
	if(label == 0 && this->unit_out.value > 0.3) {
	  pull = -1; // the score was too high for a positive example, pull down
	}

	this->circuit3->backward(this->circuit3, pull);
	this->circuit2->backward(this->circuit2, pull);
	this->circuit1->backward(this->circuit1, pull);
}

void parameterUpdate(SVM *this) {
	float step_size = 0.01;
	this->a1.value += step_size * this->a1.grad;
	this->b1.value += step_size * this->b1.grad;
	this->c1.value += step_size * this->c1.grad;

	this->a2.value += step_size * this->a2.grad;
	this->b2.value += step_size * this->b2.grad;
	this->c2.value += step_size * this->c2.grad;

	this->a3.value += step_size * this->a3.grad;
	this->b3.value += step_size * this->b3.grad;
	this->c3.value += step_size * this->c3.grad;
}

void learnFrom(SVM *this, Unit *x, Unit *y, int label) {
	this->forward(this, x, y);
	this->backward(this, label);
	this->parameterUpdate(this);
}

void init_SVM(SVM *svm) {
	svm->circuit1 = new_Circuit();
	svm->circuit2 = new_Circuit();
	svm->circuit3 = new_Circuit();
	svm->forward = forward_SVM;
	svm->backward = backward_SVM;
	svm->parameterUpdate = parameterUpdate;
	svm->learnFrom = learnFrom;
	
	svm->a1.value = (float)rand()/RAND_MAX;
	svm->a1.grad = 0;
	svm->b1.value = (float)rand()/RAND_MAX;
	svm->b1.grad = 0;
	svm->c1.value = (float)rand()/RAND_MAX;
	svm->c1.grad = 0;

	svm->a2.value = (float)rand()/RAND_MAX;
	svm->a2.grad = 0;
	svm->b2.value = (float)rand()/RAND_MAX;
	svm->b2.grad = 0;
	svm->c2.value = (float)rand()/RAND_MAX;
	svm->c2.grad = 0;

	svm->a3.value = (float)rand()/RAND_MAX;
	svm->a3.grad = 0;
	svm->b3.value = (float)rand()/RAND_MAX;
	svm->b3.grad = 0;
	svm->c3.value = (float)rand()/RAND_MAX;
	svm->c3.grad = 0;
}

float getRandomArbitrary(float min, float max) {
	return ((float)rand()/RAND_MAX) * (max - min) + min;
}

float evalTrainingAccuracy(SVM *svm, int (*data)[2], int *labels, int len) {
	float num_correct = 0;
	Unit x;
	Unit y;
	int true_label;
	for(int i = 0; i < len; i++) {
		x.value = data[i][0];
		y.value = data[i][1];
		x.grad = 0;
		y.grad = 0;
		true_label = labels[i];
		int predicted_label = svm->forward(svm, &x, &y)->value > 0.8 ? 1 : 0;
		if(predicted_label == true_label) {
			num_correct++;
		}
	}
	return num_correct / len;
};

void Random_Test_XOR(SVM *svmXOR) {
	int data[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
	int labels[8] = {0, 1, 1, 0};
	int num_correct = 0;
	Unit *x = malloc(sizeof(Unit));
	Unit *y = malloc(sizeof(Unit));
	x->grad = 0;
	y->grad = 0;
	int true_label;
	int TESTNUM = 1000000;
	for(int iter = 0; iter < TESTNUM; iter++) {
		int i = iter % 4;
		x->value = data[i][0] == 0 ? getRandomArbitrary(0, 0.2) : getRandomArbitrary(0.8, 1);
		y->value = data[i][1] == 0 ? getRandomArbitrary(0, 0.2) : getRandomArbitrary(0.8, 1);
		Unit *xor2 = svmXOR->forward(svmXOR, x, y);
		true_label = labels[i];
		int predicted_label = xor2->value > 0.8 ? 1 : 0;
		if(predicted_label == true_label) {
			num_correct++;
		}
		else {
			//printf("err: %d, %d %d\n", i, predicted_label, true_label);
		}
	}
	free(x); x = NULL;
	free(y); y = NULL;

	printf("XOR-GATE 隨機輸入測試：%d/%d %s\n", num_correct, TESTNUM, (num_correct == TESTNUM ? "PASSED" : "")) ;
}

int main(void) {
	srand(time(0));

	TestCircuit();

	SVM svmXOR; init_SVM(&svmXOR);
	
	Unit x = { .value = (float)rand()/RAND_MAX, 0 };
	Unit y = { .value = (float)rand()/RAND_MAX, 0 };

	int data[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
	int labelsXOR[4] = {0, 1, 1, 0};
	int *labelList[] = {labelsXOR};
	SVM *svmList[] = {&svmXOR};
	char *nameList[] = {"svmXOR"};

	for(int svmCnt=0; svmCnt<1; ++svmCnt) {
		SVM *svm = svmList[svmCnt];
		int *labels = labelList[svmCnt];
		for(int iter=0; iter<100000; ++iter) {
			int i = rand() % 4;
			x.value = data[i][0];
			x.grad = 0;
			y.value = data[i][1];
			y.grad = 0;
			x.value = x.value == 0 ? getRandomArbitrary(0, 0.3) : getRandomArbitrary(0.7, 1);
			y.value = y.value == 0 ? getRandomArbitrary(0, 0.3) : getRandomArbitrary(0.7, 1);
			svm->learnFrom(svm, &x, &y, labels[i]);

			if(0 && iter % 250 == 0) {
				float errRate = evalTrainingAccuracy(svm, data, labels, 4);
				printf("training accuracy at iter %d: %f\n", iter, errRate);
				printf("%f %f %f\n", svm->a1.value, svm->b1.value, svm->c1.value);
				printf("%f %f %f\n", svm->a2.value, svm->b2.value, svm->c2.value);
				printf("%f %f %f\n", svm->a3.value, svm->b3.value, svm->c3.value);
			}
		}
		printf("%s\n", nameList[svmCnt]);
		float errRate = evalTrainingAccuracy(svm, data, labels, 4);
		printf("training accuracy: %f\n", errRate);
		printf("%f, %f, %f\n", svm->a1.value, svm->b1.value, svm->c1.value);
		printf("%f, %f, %f\n", svm->a2.value, svm->b2.value, svm->c2.value);
		printf("%f, %f, %f\n", svm->a3.value, svm->b3.value, svm->c3.value);
		printf("--------\n");
	}

	Random_Test_XOR(&svmXOR);
}