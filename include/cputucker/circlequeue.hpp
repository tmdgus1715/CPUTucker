#include <mutex>
#include <condition_variable>
#include <stdexcept>

template<typename T>
class CircularQueue {
private:
    int front, rear, size;
    T* queue;
    int capacity;

public:
    CircularQueue(int capacity) : front(0), rear(0), size(0), capacity(capacity) {
        queue = new T[capacity];
    }

    ~CircularQueue() {
        delete[] queue;
    }

    CircularQueue(const CircularQueue&) = delete;
    CircularQueue& operator=(const CircularQueue&) = delete;

    void enqueue(const T& item) {
        queue[rear] = item;
        rear = (rear + 1) % capacity;
        size++;
    }

    bool dequeue(T& item) {
        item = queue[front];
        front = (front + 1) % capacity;
        size--;
        return true;
    }

    bool isEmpty() const {
        return size == 0;
    }

    bool isFull() const {
        return size == capacity;
    }

    int getSize() const {
        return size;
    }

    void resize(int newCapacity) {
        T* newQueue = new T[newCapacity];
        int i = 0;
        while (!isEmpty()) {
            dequeue(newQueue[i]);
            i++;
        }
        delete[] queue;
        queue = newQueue;
        front = 0;
        rear = i;
        capacity = newCapacity;
    }
};
