# Opções de compilação e ligação
CFLAGS = -Wall -g -O3
LDLIBS = -lm -lpthread

# Lista de objetos
objs = main.o

# Alvo padrão
all: acharKMenoresSeq

# Regra para criar o executável
acharKMenoresSeq: $(objs)
	gcc $(CFLAGS) -o acharKMenoresSeq $(objs)

# Regra para compilar os objetos
%.o: %.c
	gcc $(CFLAGS) -c $< -o $@

# Regra para limpar arquivos temporários
clean:
	-rm -f $(objs) *~

# Regra para limpar tudo (objetos e executável)
purge: clean
	-rm -f acharKMenoresSeq