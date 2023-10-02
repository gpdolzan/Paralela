# Opções de compilação e ligação
CFLAGS = -Wall -g -O3
LDLIBS = -lm -lpthread

# Lista de objetos
objs_acharKMenores = acharKMenores.o chrono.o
objs_media = media.o

# Alvo padrão
all: acharKMenores media

# Regra para criar o executável
acharKMenores: $(objs_acharKMenores)
	gcc $(CFLAGS) -o acharKMenores $(objs_acharKMenores) $(LDLIBS)

media: $(objs_media)
	gcc $(CFLAGS) -o media $(objs_media)

# Regra para compilar os objetos
%.o: %.c
	gcc $(CFLAGS) -c $< -o $@

# Regra para limpar arquivos temporários
clean:
	-rm -f $(objs_acharKMenores) *~
	-rm -f $(objs_media) *~

# Regra para limpar tudo (objetos e executável)
purge: clean
	-rm -f acharKMenores media