from os import path
from json import load
from re import compile
from statistics import mean

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

def contar_caracteres(tokenizado):
    acumulador = 0
    for token in tokenizado:
        acumulador += len(token)

    return acumulador

def contar_palavras(tokenizado):
    return len(
        remover_pontuacao(tokenizado)
    )

def frequencia_palavras(tokenizado):
    tokenizado = remover_pontuacao
    return FreqDist(tokenizado)

def frequencia_emails(lista_emails):
    return FreqDist(lista_emails)

def contar_termos_mais_comuns(freq_dist, quantidade = None):
    if quantidade == None:
        return freq_dist.most_common(freq_dist.B())
    else:
        return freq_dist.most_common(quantidade)

def tokenizar_frases(texto, linguagem, termos_banidos):
    frases = sent_tokenize(texto, linguagem)
    frases = remover_termos(frases, termos_banidos)

    return frases

def tokenizar_palavras(texto, linguagem, termos_banidos, stop_words):
    palavras = word_tokenize(texto, linguagem)
    palavras = remover_termos(palavras, termos_banidos)
    palavras = remover_case(palavras)
    palavras = remover_stopwords(palavras, stop_words)

    return palavras

def analisar_sentimento(frases):
    s = SentimentIntensityAnalyzer()
    pontuacoes = [s.polarity_scores(frase)["compound"] for frase in frases]
    pontuacao = mean(pontuacoes)

    classificao = "False"
    if pontuacao > 0.05:
        classificao = True

    return {'classificao_positiva': classificao, 'pontuacao': pontuacao}

config = carregar_configuracoes("src/config.json")

TEXTO_ANALISE = path.abspath(config["texto_analise"])
LINGUAGEM = config["linguagem"]
ENCODING = config["encoding"]
CARACTERES_BANIDOS = config["caracteres_banidos"]
STOPWORDS = stopwords.words(LINGUAGEM)

texto = carregar_texto(TEXTO_ANALISE, ENCODING)
tokens_frase = tokenizar_frases(texto, LINGUAGEM, CARACTERES_BANIDOS)
tokens_palavra = tokenizar_palavras(texto, LINGUAGEM, CARACTERES_BANIDOS, STOPWORDS)
emails = extrair_enderecos_email(tokens_frase)

#1. Usernames dos endereços de e-mails presentes no texto
print("\nOs usernames dos endereços de e-mail presentes no texto são: ")
print(extrair_username_email(emails))

#2. Domínios dos endereços de e-mails e quantas vezes cada um aparece no texto
print("\nOs domínios dos endereços de e-mail presentes no texto, bem como sua quantidade de ocorrências, são: ")
print()