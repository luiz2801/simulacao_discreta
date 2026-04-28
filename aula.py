from collections import Counter
import statistics as st
import math as m

def extrat_outlier(lista):
    q1 = st.quantiles(lista)[0]
    q3 = st.quantiles(lista)[2]
    a = q3 - q1
    print("q1:", q1, ", q3:", q3, ", A:", a)

    for i in lista:
        if i < (q1 - 3*a) or i > (q3 + 3*a):
            print("removendo o outlier:", i)
            lista.remove(i)
    return lista

def normal_pdf(x, mu, sigma):
    coef = 1 / (sigma * m.sqrt(2 * m.pi))
    expoente = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coef * m.exp(expoente)

def criar_tabela_ks(lista):
    tabela_ks = []
    lista.sort()
    media_inv = st.mean(lista)
    dp = st.pstdev(lista)
    print(media_inv, dp)
    valores_unicos = sorted(set(lista))
    for i in range(len(valores_unicos)):
        valor = valores_unicos[i]
        fatn = normal_pdf(valor, media_inv, dp)
        print("olha aqui", fatn)
        if len(tabela_ks) == 0:
            foa = lista.count(valor)
        else:
            foa = lista.count(valor) + tabela_ks[i-1][2]
        faon = foa/len(lista)
        tabela_ks.append([valor, lista.count(valor), foa, faon, fatn, abs(faon - fatn)])
    for i in tabela_ks:
        print(i)
    return tabela_ks

# def cria_tabela_ks_nova(lista):


lista = [11, 5, 2, 0, 9, 9, 1, 5, 1, 3,
3, 3, 7, 4, 12, 8, 5, 2, 6, 1,
11, 1, 2, 4, 2, 1, 3, 9, 0, 10,
3, 3, 1, 5, 18, 4, 22, 8, 3, 0,
8, 9, 2, 3, 12, 1, 3, 1, 7, 5,
14, 7, 7, 28, 1, 3, 2, 11, 13, 2,
0, 1, 6, 12, 15, 0, 6, 7, 19, 1,
1, 9, 1, 5, 3, 17, 10, 15, 43, 2,
6, 1, 13, 13, 19, 10, 9, 20, 19, 2,
27, 5, 20, 5, 10, 8, 2, 3, 1, 1,
4, 3, 6, 13, 10, 9, 1, 1, 3, 9,
9, 4, 0, 3, 6, 3, 27, 3, 18, 4,
6, 0, 2, 2, 8, 4, 5, 1, 4, 18,
1, 0, 16, 20, 2, 2, 2, 12, 28, 0,
7, 3, 18, 12, 3, 2, 8, 3, 19, 12,
5, 4, 6, 0, 5, 0, 3, 7, 0, 8,
8, 12, 3, 7, 1, 3, 1, 3, 2, 5,
4, 9, 4, 12, 4, 11, 9, 2, 0, 5,
8, 24, 1, 5, 12, 9, 17, 728, 12, 6,
4, 3, 5, 7, 4, 4, 4, 11, 3, 8]

dados_filt = extrat_outlier(lista)
tabela_ks = criar_tabela_ks(dados_filt)