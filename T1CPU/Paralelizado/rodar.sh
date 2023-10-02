#!/bin/bash

# Access the arguments passed to the script

# Check if the number of arguments is correct
if [[ $# -ne 4 ]]; then
    echo "Usage: ./rodar.sh <nTotalElements> <k> <nRepeticoesPrograma> <nomeArquivoDeLog>"
    # Stop the script
    return
fi
NELEMENTOS=$1
K=$2
NREPETICOES=$3
ARQLOG=$4

TIME_ARGS=""
THROUGHPUT_ARGS=""

# Check if 'acharKMenores' executable exists
if [[ ! -x "./acharKMenores" ]]; then
    # If it doesn't, attempt to build it using 'make'
    echo "Building 'acharKMenores' and 'media' using make..."
    make > /dev/null 2>&1
fi

# Check again if 'acharKMenores' is now available and executable
# Initialize arrays to hold time and throughput results
declare -a TIME_ARRAY
declare -a THROUGHPUT_ARRAY

# Zero arrays
TIME_ARRAY=()
THROUGHPUT_ARRAY=()

echo "Numero de repeticoes: $NREPETICOES" > $ARQLOG
echo "Numero de elementos: $NELEMENTOS" >> $ARQLOG
echo "Numero de elementos a serem encontrados: $K" >> $ARQLOG

if [[ -x "./acharKMenores" ]]; then
    for i in {1..8}; do
        echo "Running 'acharKMenores' $NREPETICOES times with $i threads..."
        for j in $(seq 1 $NREPETICOES); do
            OUTPUT=$(./acharKMenores $NELEMENTOS $K $i)
            TIME=$(awk '/Tempo:/ {print $2}' <<< "$OUTPUT")
            THROUGHPUT=$(awk '/Throughput:/ {print $2}' <<< "$OUTPUT")

            TIME_ARRAY+=("$TIME")
            THROUGHPUT_ARRAY+=("$THROUGHPUT")
        done

        echo "Numero de Threads: $i" >> $ARQLOG
        echo "TIME_ARRAY:${TIME_ARRAY[*]}" >> $ARQLOG
        echo "THROUGHPUT_ARRAY:${THROUGHPUT_ARRAY[*]}" >> $ARQLOG
        echo "Media tempo de execucao: $(./media "$NREPETICOES" "${TIME_ARRAY[@]}")" >> $ARQLOG
        echo "Media vazao: $(./media "$NREPETICOES" "${THROUGHPUT_ARRAY[@]}")" >> $ARQLOG
        echo "" >> $ARQLOG

        # Reset the arrays
        TIME_ARRAY=()
        THROUGHPUT_ARRAY=()
    done
else
    echo "Failed to build or execute 'acharKMenores'."
fi

echo "Excluindo arquivos fonte e executável..."
make purge > /dev/null 2>&1