//see: http://karpathy.github.io/neuralnets/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>

typedef struct {
	char value;
	char grad;
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
	//this->utop.value = (char)(((int)u0->value * u1->value)/(32));
	//this->utop.value = u0->value * u1->value;
	this->utop.value = (u0->value * u1->value)/32;
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
	char (* ReLu)(char x);
	Unit (*(*forward)(struct ReLuGate *this, Unit *u0));
	void (*backward)(struct ReLuGate *this);
} ReLuGate;

char ReLu(char x) {
	return x > 32 ? 32 : x > 0 ? x : 0;
}

Unit* forward_ReLuGate(ReLuGate *this, Unit *u0) {
	this->u0 = u0;
	this->utop.value = this->ReLu(u0->value);
	this->utop.grad = 0;
	return &this->utop;
}

void backward_ReLuGate(ReLuGate *this) {
	char s = this->ReLu(this->u0->value);
	this->u0->grad += (s > 0 ? 1 : 0) * this->utop.grad;
}

ReLuGate* new_ReLuGate() {
	ReLuGate *reluGate = malloc(sizeof(ReLuGate));
	reluGate->ReLu = ReLu;
	reluGate->forward = forward_ReLuGate;
	reluGate->backward = backward_ReLuGate;
	return reluGate;
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
	void (*backward)(struct Circuit *this, char gradient_top);
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

void backward_Circuit(struct Circuit *this, char gradient_top) {
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

	Unit x = { .value = 10, 0 };
	Unit y = { .value = 20, 0 };
	Unit a = { .value = 30, 0 };
	Unit b = { .value = 40, 0 };
	Unit c = { .value = 50, 0 };
	
	Unit* unit_out = circuit->forward(circuit, &x, &y, &a, &b, &c);

	// ax + by + c = 0.61
	assert((int)(unit_out->value*(32)) == 61);

	free(circuit);

	printf("TestCircuit2 [passed]\n");
}


void TestCircuit() {
	printf("TestCircuit: ReLu\n");

	Circuit *circuit = new_Circuit();

	Unit a = { .value = (32)*0.1, 0 };
	Unit b = { .value = (32)*0.2, 0 };
	Unit c = { .value = (32)*0.3, 0 };
	Unit x = { .value = (32)*0.1, 0 };
	Unit y = { .value = (32)*0.3, 0 };
	
	Unit* unit_out = circuit->forward(circuit, &x, &y, &a, &b, &c);

	printf("s: %d %d\n", unit_out->value, unit_out->grad);
	circuit->backward(circuit, 1.0);
	printf("s: %d %d\n", unit_out->value, unit_out->grad);
	printf("a: %d %d\n", a.value, a.grad);

	char step_size = 1;
	a.value += step_size * a.grad;
	b.value += step_size * b.grad;
	c.value += step_size * c.grad;
	x.value += step_size * x.grad;
	y.value += step_size * y.grad;
	unit_out = circuit->forward(circuit, &x, &y, &a, &b, &c);
	printf("s: %d\n", unit_out->value);

	assert(unit_out->value == 18);

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

	if(label == 1 && this->unit_out.value < 0.7*(32)) { 
	  pull = 1; // the score was too low: pull up
	}
	if(label == 0 && this->unit_out.value > 0.3*(32)) {
	  pull = -1; // the score was too high for a positive example, pull down
	}

	this->circuit3->backward(this->circuit3, pull);
	this->circuit2->backward(this->circuit2, pull);
	this->circuit1->backward(this->circuit1, pull);
}

void parameterUpdate(SVM *this) {
	char step_size = 1;
	this->a1.value += step_size * this->a1.grad; this->a1.value %= 128;
	this->b1.value += step_size * this->b1.grad; this->b1.value %= 128;
	this->c1.value += step_size * this->c1.grad; this->c1.value %= 128;

	this->a2.value += step_size * this->a2.grad; this->a2.value %= 128;
	this->b2.value += step_size * this->b2.grad; this->b2.value %= 128;
	this->c2.value += step_size * this->c2.grad; this->b2.value %= 128;

	this->a3.value += step_size * this->a3.grad; this->a3.value %= 128;
	this->b3.value += step_size * this->b3.grad; this->b3.value %= 128;
	this->c3.value += step_size * this->c3.grad; this->c3.value %= 128;
}

void learnFrom(SVM *this, Unit *x, Unit *y, int label) {
	this->forward(this, x, y);
	this->backward(this, label);
	this->parameterUpdate(this);
}

char getRandomArbitrary(float min, float max) {
	//printf("RANDOM: %d\n",  (int)((32) * ((((float)rand()/RAND_MAX)) * (max - min) + min)));
	return (char)((32) * ((((float)rand()/RAND_MAX)) * (max - min) + min));
}

void init_SVM(SVM *svm) {
	svm->circuit1 = new_Circuit();
	svm->circuit2 = new_Circuit();
	svm->circuit3 = new_Circuit();
	svm->forward = forward_SVM;
	svm->backward = backward_SVM;
	svm->parameterUpdate = parameterUpdate;
	svm->learnFrom = learnFrom;
	
	svm->a1.value = getRandomArbitrary(0, 1);
	svm->a1.grad = 0;
	svm->b1.value = getRandomArbitrary(0, 1);
	svm->b1.grad = 0;
	svm->c1.value = getRandomArbitrary(0, 1);
	svm->c1.grad = 0;

	svm->a2.value = getRandomArbitrary(0, 1);
	svm->a2.grad = 0;
	svm->b2.value = getRandomArbitrary(0, 1);
	svm->b2.grad = 0;
	svm->c2.value = getRandomArbitrary(0, 1);
	svm->c2.grad = 0;

	svm->a3.value = getRandomArbitrary(0, 1);
	svm->a3.grad = 0;
	svm->b3.value = getRandomArbitrary(0, 1);
	svm->b3.grad = 0;
	svm->c3.value = getRandomArbitrary(0, 1);
	svm->c3.grad = 0;
}

float evalTrainingAccuracy(SVM *svm, char (*data)[2], char *labels, char len) {
	float num_correct = 0;
	Unit x;
	Unit y;
	char true_label;
	for(int i = 0; i < len; i++) {
		x.value = data[i][0]*(32);
		y.value = data[i][1]*(32);
		x.grad = 0;
		y.grad = 0;
		true_label = labels[i];
		char predicted_label = svm->forward(svm, &x, &y)->value > 0.7*(32) ? 1 : 0;
		if(predicted_label == true_label) {
			num_correct++;
		}
	}
	return num_correct / len;
};

int Random_Test_XOR(SVM *svmXOR, char (*data)[2], char *labels, char len) {
	int num_correct = 0;
	Unit *x = malloc(sizeof(Unit));
	Unit *y = malloc(sizeof(Unit));
	x->grad = 0;
	y->grad = 0;
	char true_label;
	int TESTNUM = 100000;
	for(int iter = 0; iter < TESTNUM; iter++) {
		int i = iter % 4;
		//x->value = data[i][0]*32; //== 0 ? getRandomArbitrary(0, 0.2) : getRandomArbitrary(0.8, 1);
		//y->value = data[i][1]*32; //== 0 ? getRandomArbitrary(0, 0.2) : getRandomArbitrary(0.8, 1);
		x->value = data[i][0] == 0 ? getRandomArbitrary(0, 0.3) : getRandomArbitrary(0.7, 1);
		y->value = data[i][1] == 0 ? getRandomArbitrary(0, 0.3) : getRandomArbitrary(0.7, 1);
		Unit *xor2 = svmXOR->forward(svmXOR, x, y);
		true_label = labels[i];
		//printf("xor2->value: %f\n", xor2->value/(32.0));
		char predicted_label = xor2->value > 0.7*(32) ? 1 : 0;
		if(predicted_label == true_label) {
			num_correct++;
		}
		else {
			//printf("err: %d, %d %d %f\n", i, predicted_label, true_label, xor2->value/(32.0));
		}
	}
	free(x); x = NULL;
	free(y); y = NULL;

	printf("XOR-GATE 隨機輸入測試：%d/%d %s\n", num_correct, TESTNUM, (num_correct == TESTNUM ? "PASSED" : "")) ;
	return (num_correct == TESTNUM);
}

int main(void) {
	srand(time(0));

	TestCircuit();

	SVM svmXOR; init_SVM(&svmXOR);
	
	Unit x = { .value = ((float)rand()/RAND_MAX)*(32), 0 };
	Unit y = { .value = ((float)rand()/RAND_MAX)*(32), 0 };

	char data[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
	char labelsXOR[4] = {0, 1, 1, 0};
	char *labelList[] = {labelsXOR};
	SVM *svmList[] = {&svmXOR};
	char *nameList[] = {"svmXOR"};
	int failcnt = 0;
	int totalFailCnt = 0;
	int totaltrial = 0;
	int maxFailCnt = 0;
	int minFailCnt = 9999999;
	for(totaltrial=0; totaltrial<10; ++totaltrial) {
		failcnt = 0;
		do {
			init_SVM(&svmXOR);
			for(int svmCnt=0; svmCnt<1; ++svmCnt) {
				SVM *svm = svmList[svmCnt];
				char *labels = labelList[svmCnt];
				for(int iter=0; iter<100000; ++iter) {
					int i = rand() % 4;
					x.value = data[i][0];
					x.grad = 0;
					y.value = data[i][1];
					y.grad = 0;
					x.value = x.value == 0 ? getRandomArbitrary(0, 0.3) : getRandomArbitrary(0.7, 1);
					y.value = y.value == 0 ? getRandomArbitrary(0, 0.3) : getRandomArbitrary(0.7, 1);
					//printf("x=%d y=%d\n", x.value, y.value);
					svm->learnFrom(svm, &x, &y, labels[i]);

					if(0 && iter % 250 == 0) {
						float errRate = evalTrainingAccuracy(svm, data, labels, 4);
						printf("training accuracy at iter %d: %f\n", iter, errRate);
						printf("%d %d %d\n", svm->a1.value, svm->b1.value, svm->c1.value);
						printf("%d %d %d\n", svm->a2.value, svm->b2.value, svm->c2.value);
						printf("%d %d %d\n", svm->a3.value, svm->b3.value, svm->c3.value);
					}
				}
				printf("%s\n", nameList[svmCnt]);
				float errRate = evalTrainingAccuracy(svm, data, labels, 4);
				printf("training accuracy: %f\n", errRate);
				//printf("%d, %d, %d\n", svm->a1.value, svm->b1.value, svm->c1.value);
				//printf("%d, %d, %d\n", svm->a2.value, svm->b2.value, svm->c2.value);
				//printf("%d, %d, %d\n", svm->a3.value, svm->b3.value, svm->c3.value);
				//printf("--------\n");
				printf("%4d/32=%+.3f, %4d/32=%+.3f, %4d/32=%+.3f\n", svm->a1.value, svm->a1.value/32.0, svm->b1.value, svm->b1.value/32.0, svm->c1.value, svm->c1.value/32.0);
				printf("%4d/32=%+.3f, %4d/32=%+.3f, %4d/32=%+.3f\n", svm->a2.value, svm->a2.value/32.0, svm->b2.value, svm->b2.value/32.0, svm->c2.value, svm->c2.value/32.0);
				printf("%4d/32=%+.3f, %4d/32=%+.3f, %4d/32=%+.3f\n", svm->a3.value, svm->a3.value/32.0, svm->b3.value, svm->b3.value/32.0, svm->c3.value, svm->c3.value/32.0);
				printf("--------\n");
			}
			failcnt += 1;
		}
		while(!Random_Test_XOR(&svmXOR, data, labelsXOR, 4));
		printf("failcnt: %d\n", failcnt);
		totalFailCnt += failcnt;
		if(failcnt < minFailCnt)
			minFailCnt = failcnt;
		if(failcnt > maxFailCnt)
			maxFailCnt = failcnt;
	}
	printf("平均失敗次數：%.f, 最多：%d, 最少：%d\n", totalFailCnt / 10.0, maxFailCnt, minFailCnt);
}