
# Problema 1 - Estimando o consumo de veículos

* Minicurso _Machine Learning -- Hands on com Python_
* Samsung Ocean Manaus
* Facilitadora: Elloá B. Guedes 
* Repositório: http://bit.ly/mlpython
* Nome:
* Email: 

### Bibliotecas

Por hábito, a primeira célula do notebook costuma ser reservada para importação de bibliotecas.
A cada biblioteca nova acrescida, é necessário executar a célula para atualização e correta execução.


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
```

### Abertura do Dataset

Abra o dataset e visualize o seu cabeçalho, isto é, os primeiros exemplos nele contidos.
Isto é útil para checar se a importação foi realizada de maneira adequada e se a disposição dos dados está de acordo para os próximos passos do trabalho.


```python
df = pd.read_csv("autompg.csv",sep = ";")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>modelyear</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504.0</td>
      <td>12.0</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8.0</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693.0</td>
      <td>11.5</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8.0</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436.0</td>
      <td>11.0</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8.0</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433.0</td>
      <td>12.0</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8.0</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449.0</td>
      <td>10.5</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>ford torino</td>
    </tr>
  </tbody>
</table>
</div>



### Conhecendo o dataset

Para praticar conceitos relativos à exploração do conjunto de dados, utilize as células a seguir para prover respostas para as seguintes perguntas:

1. Quantos exemplos há no dataset?
2. Quais os atributos existentes no dataset?
3. Quais os nomes dos carros existentes no dataset?
4. Quais as características do 'chevrolet camaro'?
5. Qual a média de consumo, em galões por litro, dos carros existentes no dataset?


```python
len(df)
```




    406




```python
df.columns
```




    Index(['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
           'acceleration', 'modelyear', 'origin', 'name'],
          dtype='object')




```python
df['name']
```




    0              chevrolet chevelle malibu
    1                      buick skylark 320
    2                     plymouth satellite
    3                          amc rebel sst
    4                            ford torino
    5                       ford galaxie 500
    6                       chevrolet impala
    7                      plymouth fury iii
    8                       pontiac catalina
    9                     amc ambassador dpl
    10                  citroen ds-21 pallas
    11      chevrolet chevelle concours (sw)
    12                      ford torino (sw)
    13               plymouth satellite (sw)
    14                    amc rebel sst (sw)
    15                   dodge challenger se
    16                    plymouth 'cuda 340
    17                 ford mustang boss 302
    18                 chevrolet monte carlo
    19               buick estate wagon (sw)
    20                 toyota corona mark ii
    21                       plymouth duster
    22                            amc hornet
    23                         ford maverick
    24                          datsun pl510
    25          volkswagen 1131 deluxe sedan
    26                           peugeot 504
    27                           audi 100 ls
    28                              saab 99e
    29                              bmw 2002
                         ...                
    376             chevrolet cavalier wagon
    377            chevrolet cavalier 2-door
    378           pontiac j2000 se hatchback
    379                       dodge aries se
    380                      pontiac phoenix
    381                 ford fairmont futura
    382                       amc concord dl
    383                  volkswagen rabbit l
    384                   mazda glc custom l
    385                     mazda glc custom
    386               plymouth horizon miser
    387                       mercury lynx l
    388                     nissan stanza xe
    389                         honda accord
    390                       toyota corolla
    391                          honda civic
    392                   honda civic (auto)
    393                        datsun 310 gx
    394                buick century limited
    395    oldsmobile cutlass ciera (diesel)
    396           chrysler lebaron medallion
    397                       ford granada l
    398                     toyota celica gt
    399                    dodge charger 2.2
    400                     chevrolet camaro
    401                      ford mustang gl
    402                            vw pickup
    403                        dodge rampage
    404                          ford ranger
    405                           chevy s-10
    Name: name, Length: 406, dtype: object




```python
df.loc[df['name'] == "chevrolet camaro"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>modelyear</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>400</th>
      <td>27.0</td>
      <td>4.0</td>
      <td>151.0</td>
      <td>90.0</td>
      <td>2950.0</td>
      <td>17.3</td>
      <td>82.0</td>
      <td>1.0</td>
      <td>chevrolet camaro</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.mean(df['mpg'])
```




    23.514572864321615



### Preparação dos dados

1. Existem exemplos com dados faltantes. Para fins de simplificação, elimine-os do dataset.
2. Exclua a coluna com os nomes dos carros
3. Converta mpg para km/l sabendo que: 1 mpg  = 0.425 km/l. Utilize apenas duas casas decimais nesta conversão.
4. Remova a coluna mpg e insira a coluna kml no dataset.


```python
df.dropna(inplace=True)
```


```python
df.drop(['name'],axis = 1,inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>modelyear</th>
      <th>origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504.0</td>
      <td>12.0</td>
      <td>70.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8.0</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693.0</td>
      <td>11.5</td>
      <td>70.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8.0</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436.0</td>
      <td>11.0</td>
      <td>70.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8.0</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433.0</td>
      <td>12.0</td>
      <td>70.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8.0</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449.0</td>
      <td>10.5</td>
      <td>70.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
kml = [round(x*0.425,2) for x in df["mpg"]]
```


```python
df['kml'] = kml
df.drop(['mpg'],axis = 1,inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>modelyear</th>
      <th>origin</th>
      <th>kml</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.0</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504.0</td>
      <td>12.0</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>7.65</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.0</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693.0</td>
      <td>11.5</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>6.38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436.0</td>
      <td>11.0</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>7.65</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.0</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433.0</td>
      <td>12.0</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>6.80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.0</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449.0</td>
      <td>10.5</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>7.22</td>
    </tr>
  </tbody>
</table>
</div>



### Organização dos dados para treinamento

1. Remova a coluna kml e atribua-a a uma variável Y
2. Atribua os demais valores do dataset a uma variável X
3. Efetue uma partição holdout 70/30 com o sklearn


```python
Y = df['kml']
df.drop(["kml"],axis = 1,inplace=True)
X = df
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>modelyear</th>
      <th>origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.0</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504.0</td>
      <td>12.0</td>
      <td>70.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.0</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693.0</td>
      <td>11.5</td>
      <td>70.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436.0</td>
      <td>11.0</td>
      <td>70.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.0</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433.0</td>
      <td>12.0</td>
      <td>70.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.0</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449.0</td>
      <td>10.5</td>
      <td>70.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Necessário importar: from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
```

### Treinamento de um modelo de regressão linear

1. Importe o modelo da biblioteca sklearn
2. Instancie o modelo com parâmetros padrão (default)
3. Execute o algoritmo de treinamento com os dados de treino


```python
regr = LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



### Teste do modelo

Vamos observar a saída do modelo para um exemplo individual existente nos dados de treino:
* Atributos preditores: X_test[2:3]
* Atributo alvo: Y_test.iloc[2]
* Qual o resultado previsto para o modelo, dados estes atributos preditores?


```python
teste1 = X_test[2:3]
resultado = regr.predict(teste1)
print(resultado,Y_test.iloc[2])
```

    [12.23482843] 13.17


### Teste do modelo

1. Obtenha o erro médio quadrático para todos os dados de teste
 * Efetue a importação de mean_squared_error do pacote sklearn.metrics
 * Trata-se do somatório do quadrado das diferenças entre valores previstos pelo modelo e observados na prática
 * Quanto mais próximo de zero, melhor este resultado
2. Obtenha o r^2 para os dados de teste
 * Efetue a importação de r2_score do pacote sklearn.metrics
 * Trata-se de um valor no intervalo [0,1]
 * Quanto mais próximo de 1, melhor é o modelo


```python
Y_predito = regr.predict(X_test)
mse = mean_squared_error(Y_predito,Y_test)
mse
```




    2.3467645499015197




```python
r2 = r2_score(Y_predito,Y_test)
r2
```




    0.7178368636351276



### Obtendo e visualizando os resíduos

Uma maneira muito comum de visualizarmos o quão bom certo modelo é para aprender determinados padrões dá-se por meio da visualização dos resíduos, isto é, da diferença entre os valores previstos e observados. Adapte o código a seguir para calcular os resíduos produzidos pelo seu modelo.


```python
residuos = []
for (x,y) in zip(Y_test,Y_predito):
    residuos.append((x-y)**2)
residuos
```




    [0.15983751942609256,
     0.5477253341534122,
     0.8745458627157038,
     3.4016512086693247,
     1.79385769143608,
     1.655655300327717,
     0.11282331390144802,
     0.49997285239461897,
     1.5863636946995898,
     1.8508140907081765,
     5.247065117838299,
     10.268433785640894,
     0.49209594566266307,
     0.2424416916038056,
     0.0060830472814778075,
     0.886774072939592,
     4.869870219222626,
     2.918081276094515,
     0.8986861205012131,
     2.156089349140755,
     0.00014746155893131718,
     0.20399562242664449,
     2.934517266711463,
     0.6215776557515488,
     2.552322237334607,
     4.729032943931411,
     0.9946020896605926,
     1.2978426399095098,
     0.27971264033155585,
     2.864819126040625,
     0.0014021511448079637,
     16.30738859286149,
     1.6171563107845675,
     3.0572122585479316e-05,
     4.187500903818978,
     6.406369440607882,
     1.508846358444241,
     0.5099213793466693,
     18.4710774563034,
     0.41460806533927247,
     2.4990292345223555,
     0.1720717910835699,
     2.0438299066119394,
     0.4309622592168786,
     0.6874826503351446,
     2.0426547658290266,
     7.312434095470637,
     0.015177548491369618,
     0.004127489445743562,
     0.61706897003951,
     3.166759441354219,
     0.5438561132437669,
     0.4292140389594566,
     0.17991920792860158,
     5.932166757208845,
     0.7729174103806807,
     0.1700435857233917,
     0.05564218783631551,
     7.0040838994159085,
     0.8057393055835091,
     1.1664134706969975,
     0.4857358247114144,
     0.14805306332020396,
     0.0409593499214001,
     0.08527013376204398,
     0.8932348118725043,
     0.03803221126646245,
     19.05971780874447,
     1.183144655233895,
     10.146030291403935,
     0.7492193532517428,
     0.3212596132125894,
     2.623143217715548,
     2.186055775899068,
     0.9459383830365404,
     0.3822147319024309,
     1.0952345277434197,
     3.0863011424428555e-05,
     1.784942991380229,
     1.486273372836293,
     0.20728883177107246,
     0.0054118052975959275,
     2.4419399037130955,
     0.13817431054067783,
     16.35862731606127,
     6.584267082192764,
     0.13559266442374543,
     0.0023100959768103707,
     0.01898055329618769,
     0.25562498477392254,
     0.20332503310992725,
     0.0023845834409507425,
     0.6194878039984627,
     0.6307161609497937,
     0.00012677454728716027,
     2.7307325944011422,
     0.03848688515779671,
     0.15894500574196785,
     0.026948479471865197,
     1.0231630358116062,
     0.37244487114979474,
     2.1538402741672886,
     1.4612154837622102,
     0.7945362807083058,
     2.145593839294284,
     0.16244240961925194,
     2.076463754190866e-06,
     4.602372262256416,
     32.94596794166796,
     1.1476198557727533,
     0.014170241523763286,
     0.20629609623931486,
     2.9657454385044764,
     4.182835998065352,
     2.226551645148142,
     0.22755705400483658,
     0.07196191434238049,
     3.576705782618817]




```python
x = [0,int(max(Y_test))]
y = [0,0]
plt.plot(x,y,linewidth=3)
plt.plot(Y_test,residuos,'ro')
plt.ylabel('Residuos')
plt.xlabel('kml')
plt.show()
```


![png](Machine%20Learning%20hands-on%20Python%20-%20Problema%201%20--%20Novo%20Gabarito_files/Machine%20Learning%20hands-on%20Python%20-%20Problema%201%20--%20Novo%20Gabarito_28_0.png)


### Testando K-Vizinhos Mais Próximos

1. Efetue o treinamento do K-Vizinhos mais Próximos, considerando k =5
2. Obtenha o MSE para o conjunto de testes
3. Comparando com os resultados obtidos anteriormente, qual modelo melhor endereça esse problema?


```python
# Necessário importar: from sklearn.neighbors import KNeighborsRegressor
kviz = KNeighborsRegressor(n_neighbors=5)
kviz.fit(X_train,Y_train)
```




    KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
              metric_params=None, n_jobs=1, n_neighbors=3, p=2,
              weights='uniform')




```python
Y_preditoKViz = kviz.predict(X_test)
msekviz = mean_squared_error(Y_preditoKViz,Y_test)
msekviz
```




    4.13158088512241


