from os import path
from json import load
from re import compile
from statistics import mean
from enum import Enum

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist

from Lib.LeIA.leia import SentimentIntensityAnalyzer

def carregar_configuracoes(arquivo_origem):
    abs_path = path.abspath(arquivo_origem)
    with open(abs_path) as arquivo_config:
        config_json = load(arquivo_config)

    return config_json

def carregar_texto(arquivo_origem, encoding):
    abs_path = path.abspath(arquivo_origem)
    f = open(abs_path, encoding = encoding, mode = "r")
    return f.read()

def extrair_enderecos_email(tokenizado):
    lista_emails = []
    regex = compile(r'[\w\.-]+@[\w\.-]+')
    for token in tokenizado:
        resultado_regex = regex.findall(token)
        if resultado_regex:
            email = remover_pontuacao_fim_termo(resultado_regex[0])
            lista_emails.append(email)
    
    return lista_emails

def remover_pontuacao_fim_termo(termo):
    if not termo[-1].isalnum():
        termo = termo[:-1]

    return termo

def extrair_username_email(lista_emails):
    usernames = []
    for email in lista_emails:
        usernames.append(email.split('@')[0])

    return usernames

def extrair_dominio_email(lista_emails):
    dominios = []
    for email in lista_emails:
        dominio = email.split('@')[1]
        
        dominios.append(dominio)

    return dominios

def remover_termos(tokenizado, termos):
    novos_tokens = []
    for token in tokenizado:
        token_limpo = token
        for termo in termos:
            token_limpo = token_limpo.strip(termo)
        
        novos_tokens.append(token_limpo)

    return novos_tokens

def remover_pontuacao(tokenizado):
    return [token for token in tokenizado if token.isalnum()]

def remover_case(tokenizado):
    return [token.lower() for token in tokenizado]

def remover_stopwords(tokenizado, stop_words):
    sem_stop_words = [token for token in tokenizado if not token in stop_words]
    return sem_stop_words

def contar_tokens(tokenizado):
    return len(tokenizado)

def contar_caracteres(tokenizado, caracteres_banidos):
    acumulador = 0
    for token in tokenizado:
        if token != " " and token not in caracteres_banidos:
            acumulador += len(token)

    return acumulador

def contar_palavras(tokenizado):
    return len(
        remover_pontuacao(tokenizado)
    )

def frequencia_palavras(tokenizado):
    tokenizado = remover_pontuacao(tokenizado)
    return FreqDist(tokenizado)

def frequencia_dominios(lista_emails):
    return FreqDist(lista_emails)

def contar_termos_mais_comuns(freq_dist, quantidade = None):
    if quantidade == None:
        return freq_dist.most_common(freq_dist.B())
    else:
        return freq_dist.most_common(quantidade)

def tokenizar_frases(texto, linguagem, caracteres_banidos):
    frases = sent_tokenize(texto, linguagem)
    frases = remover_termos(frases, caracteres_banidos)

    return frases

def tokenizar_palavras(texto, linguagem, caracteres_banidos, stop_words):
    palavras = word_tokenize(texto, linguagem)
    palavras = remover_termos(palavras, caracteres_banidos)
    palavras = remover_case(palavras)
    palavras = remover_stopwords(palavras, stop_words)

    return palavras

def tokenizar_palavras_com_stopwords(texto, linguagem, caracteres_banidos):
    palavras = word_tokenize(texto, linguagem)
    palavras = remover_termos(palavras, caracteres_banidos)
    palavras = remover_case(palavras)

    return palavras

def analisar_sentimento(frases):
    s = SentimentIntensityAnalyzer()
    pontuacoes = [s.polarity_scores(frase)["compound"] for frase in frases]
    valor = mean(pontuacoes)

    sentimento = Sentimento.negativo
    if valor > 0.05:
        sentimento = Sentimento.positivo

    return {'sentimento': sentimento.name, 'valor': valor}

class Sentimento(Enum):
    negativo, positivo = range(0, 2)

config = carregar_configuracoes("src/config.json")

TEXTO_ANALISE = path.abspath(config["texto_analise"])
LINGUAGEM = config["linguagem"]
ENCODING = config["encoding"]
CARACTERES_BANIDOS = config["caracteres_banidos"]
STOPWORDS = stopwords.words(LINGUAGEM)

texto = carregar_texto(TEXTO_ANALISE, ENCODING)
tokens_frase = tokenizar_frases(texto, LINGUAGEM, CARACTERES_BANIDOS)
tokens_palavra = tokenizar_palavras(texto, LINGUAGEM, CARACTERES_BANIDOS, STOPWORDS)
tokens_palavra_com_stopwords = tokenizar_palavras_com_stopwords(texto, LINGUAGEM, CARACTERES_BANIDOS)
emails = extrair_enderecos_email(tokens_frase)

#1. Usernames dos endereços de e-mails presentes no texto
print("\n1. Os usernames dos endereços de e-mail presentes no texto são: ")
print(extrair_username_email(emails))

#2. Domínios dos endereços de e-mails e quantas vezes cada um aparece no texto
print("\n2. Os domínios dos endereços de e-mail presentes no texto, seguidos de sua quantidade de ocorrências, são: ")
dominio_emails = extrair_dominio_email(emails)
obj_freq_dominios = frequencia_dominios(dominio_emails)
print(contar_termos_mais_comuns(obj_freq_dominios))

#3. As 8 palavras mais comuns excluindo as stopwords
print("\n3. As 8 palavras mais comuns, excluindo as stopwords, seguidas de sua quantidade de ocorrências, são: ")
obj_freq_tokens_palavra = frequencia_palavras(tokens_palavra)
print(contar_termos_mais_comuns(obj_freq_tokens_palavra, 8))

#4. Sentimento do texto (positivo ou negativo) e a pontuação da previsão
print("\n4. O sentimento predonimante no texto seguido do valor de sentimento normalizado de -1 a 1 é: ")
print(analisar_sentimento(tokens_frase))

#5. Quantidade de tokens
print("\n5. A quantidade de tokens incluindo pontuação e excluindo stopwords é: ")
print(contar_tokens(tokens_palavra))

#6. Quantidade de palavras e caracteres
print("\n5. A quantidade de palavras, seguida da quantidade de caracteres sem contar os espaços, é: ")
print("Palavras: {}. Caracteres: {}".format(contar_palavras(tokens_palavra_com_stopwords), contar_caracteres(tokens_frase, CARACTERES_BANIDOS)))