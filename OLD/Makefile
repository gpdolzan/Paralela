# Opções de compilação e ligação
CFLAGS = -Wall -g -O3
LDLIBS = -lm -lpthread -ltime

# Lista de objetos
objs = main.o chrono.o

# Alvo padrão
all: acharKMenores

# Regra para criar o executável
acharKMenores: $(objs)
	gcc $(CFLAGS) -o acharKMenores $(objs)

# Regra para compilar os objetos
%.o: %.c
	gcc $(CFLAGS) -c $< -o $@

# Regra para limpar arquivos temporários
clean:
	-rm -f $(objs) *~

# Regra para limpar tudo (objetos e executável)
purge: clean
	-rm -f acharKMenores