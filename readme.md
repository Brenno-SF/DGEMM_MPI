# DGEMM Paralelo com MPI

Implementação em C da multiplicação de matrizes densas em **ponto flutuante de precisão dupla (DGEMM)** utilizando **MPI (Message Passing Interface)** para paralelização em um modelo de memória distribuída.

---

## 📘 Descrição

O programa realiza o produto matricial `C = αAB + βC` em duas versões:

- **Sequencial:** executada apenas no processo raiz (`rank 0`);
- **Paralela (MPI):** divide a matriz `A` entre os processos, replica `B` e reúne o resultado final com `MPI_Gather`.

O objetivo é comparar o tempo de execução, speedup e eficiência entre as versões sequencial e paralela.

---

## ⚙️ Funcionalidades

- Geração de matrizes aleatórias (`randomMatrix`)
- Transposição otimizada por blocos (`transpose`)
- Multiplicação matricial com blocos (`dgemm`)
- Execução paralela distribuída via MPI (`runDgemmMpi`)
- Cálculo de **Speedup**, **Eficiência** e **Diferença Relativa Máxima** entre resultados

---

## 🧩 Estrutura do Código

| Função | Descrição |
|--------|------------|
| `randomMatrix()` | Preenche uma matriz com valores aleatórios entre 0 e 1 |
| `transpose()` | Transpõe uma matriz utilizando blocos de cache (`blockSize = 64`) |
| `dgemm()` | Realiza a multiplicação de matrizes com otimização por blocos |
| `runDgemmMpi()` | Executa a versão paralela usando `MPI_Scatter`, `MPI_Allgather` e `MPI_Gather` |
| `calculateMaxDiff()` | Compara a diferença entre resultados sequencial e paralelo |

---

## 🚀 Compilação e Execução

### Compilar
```bash
mpicc -O3 -march=native -lm -o dgemm_mpi dgemm_mpi.c
```

### Executar (exemplo com 4 processos)
```bash
mpirun -np 4 ./dgemm_mpi 2048
```

O número `2048` indica o tamanho `N` da matriz quadrada (N x N).

---

## 📊 Saída Esperada

O programa imprime(exemplo):

```
Generate random matrix 1 (2048 x 2048)
Generate random matrix 2 (2048 x 2048)

Sequential time: 12.345678 s
dgemmMPI time (4 procs): 3.210987 s
Speedup with 4 procs: 3.84
Efficiency with 4 procs: 0.96
Max relative difference between sequential and MPI: 1.23e-15
```

---

## 📎 Observações

- `n` deve ser divisível pelo número de processos (`-np`).
- Requer instalação do **OpenMPI**.
- Testado com matrizes de até `4096 x 4096`.

---

## 👨‍💻 Autores

**Brenno Santos Florêncio e Mateus Soares**  
Implementação e análise de desempenho em processamento paralelo (DGEMM com MPI).
