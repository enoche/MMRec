# EVALUATING THE SOTA MODELS

we validate the effectiveness and efficiency of state-of-the-art multimodal recommendation models by conducting extensive experiments on four public datasets. Furthermore, we investigate the principal determinants of model performance, including the impact of different modality information and data split methods.

## Statistics of the evaluated datasets.
| Datasets | # Users | # Items | # Interactions |Sparsity|
|----------|--------|---------|---------|---------|
| Baby     | 19,445     | 7,050     |160,792|99.8827%|
| Sports   | 35,598      | 18,357   |296,337|99.9547%|
| FoodRec     | 61,668      | 21,874    |1,654,456|99.8774%|
| Elec     | 192,403      | 63,001     |1,689,188|99.9861%|


## Experimental Results
| Dataset                 | Model    | Recall@10          | Recall@20          | Recall@50          | NDCG@10            | NDCG@20            | NDCG@50            |
|-------------------------|----------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| **Baby**   | BPR      | 0.0357             | 0.0575             | 0.1054             | 0.0192             | 0.0249             | 0.0345             |
|                         | LightGCN | 0.0479             | 0.0754             | 0.1333             | 0.0257             | 0.0328             | 0.0445             |
|                         | VBPR     | 0.0423             | 0.0663             | 0.1212             | 0.0223             | 0.0284             | 0.0396             |
|                         | MMGCN    | 0.0378             | 0.0615             | 0.1100             | 0.0200             | 0.0261             | 0.0359             |
|                         | DualGNN  | 0.0448             | 0.0716             | 0.1288             | 0.0240             | 0.0309             | 0.0424             |
|                         | GRCN     | 0.0539             | 0.0833             | 0.1464             | 0.0288             | 0.0363             | 0.0490             |
|                         | LATTICE  | 0.0547             | 0.0850             | 0.1477             | 0.0292             | 0.0370             | 0.0497             |
|                         | BM3      | 0.0564             | 0.0883             | 0.1477             | 0.0301             | 0.0383             | 0.0502             |
|                         | SLMRec   | 0.0529             | 0.0775             | 0.1252             | 0.0290             | 0.0353             | 0.0450             |
|                         | ADDVAE   | _0.0598_ | _0.091_  | _0.1508_ | _0.0323_ | _0.0404_ | _0.0525_ |
|                         | FREEDOM  | **0.0627**    | **0.0992**    | **0.1655**    | **0.0330**    | **0.0424**    | **0.0558**    |
| **Sports**  | BPR      | 0.0432             | 0.0653             | 0.1083             | 0.0241             | 0.0298             | 0.0385             |
|                         | LightGCN | 0.0569             | 0.0864             | 0.1414             | 0.0311             | 0.0387             | 0.0498             |
|                         | VBPR     | 0.0558             | 0.0856             | 0.1391             | 0.0307             | 0.0384             | 0.0492             |
|                         | MMGCN    | 0.0370             | 0.0605             | 0.1078             | 0.0193             | 0.0254             | 0.0350             |
|                         | DualGNN  | 0.0568             | 0.0859             | 0.1392             | 0.0310             | 0.0385             | 0.0493             |
|                         | GRCN     | 0.0598             | 0.0915             | 0.1509             | 0.0332             | 0.0414             | 0.0535             |
|                         | LATTICE  | 0.0620             | 0.0953             | 0.1561             | 0.0335             | 0.0421             | 0.0544             |
|                         | BM3      | 0.0656             | 0.0980             | 0.1581             | 0.0355             | 0.0438             | 0.0561             |
|                         | SLMRec   | 0.0663             | 0.0990             | 0.1543             | 0.0365             | 0.0450             | 0.0562             |
|                         | ADDVAE   | _0.0709_ | _0.1035_ | _0.1663_ | _0.0389_    | _0.0473_ | _0.0600_ |
|                         | FREEDOM  | **0.0717**    | **0.1089**    | **0.1768**    | **0.0385** | **0.0481**    | **0.0618**    |
| **FoodRec** | BPR      | 0.0303             | 0.0511             | 0.0948             | 0.0188             | 0.0250             | 0.0356             |
|                         | LightGCN | 0.0331             | 0.0546             | 0.1003             | 0.0210             | 0.0274             | 0.0386             |
|                         | VBPR     | 0.0306             | 0.0516             | 0.0972             | 0.0191             | 0.0254             | 0.0365             |
|                         | MMGCN    | 0.0307             | 0.0510             | 0.0943             | 0.0192             | 0.0253             | 0.0359             |
|                         | DualGNN  | _0.0338_ | 0.0559             | _0.1027_ | _0.0214_ | _0.0280_ | _0.0394_ |
|                         | GRCN     | **0.0356**   | **0.0578**    | **0.1063**    | **0.0226**    | **0.0295**    | **0.0411**    |
|                         | LATTICE  | 0.0336             | _0.0560_| 0.1012             | 0.0211             | 0.0277             | 0.0388             |
|                         | BM3      | 0.0334             | 0.0553             | 0.0994             | 0.0208             | 0.0274             | 0.0381             |
|                         | SLMRec   | 0.0323             | 0.0515             | 0.0907             | 0.0208             | 0.0266             | 0.0362             |
|                         | ADDVAE   | 0.0309             | 0.0508             | 0.093              | 0.0186             | 0.0247             | 0.035              |
|                         | FREEDOM  | 0.0333             | 0.0556             | 0.1009             | 0.0212             | 0.0279             | 0.0389             |
| **Elec**    | BPR      | 0.0235             | 0.0367             | 0.0621             | 0.0127             | 0.0161             | 0.0212             |
|                         | LightGCN | 0.0363             | 0.0540             | 0.0879             | 0.0204             | 0.0250             | 0.0318             |
|                         | VBPR     | 0.0293             | 0.0458             | 0.0778             | 0.0159             | 0.0202             | 0.0267             |
|                         | MMGCN    | 0.0213             | 0.0343             | 0.0610             | 0.0112             | 0.0146             | 0.0200             |
|                         | DualGNN  | 0.0365             | 0.0542             | 0.0875             | 0.0206             | 0.0252             | 0.0319             |
|                         | GRCN     | 0.0389             | 0.0590             | 0.0970             | 0.0216             | 0.0268             | 0.0345             |
|                         | LATTICE  | -                  | -                  | -                  | -                  | -                  | -                  |
|                         | BM3      | 0.0437             | 0.0648             | 0.1021             | 0.0247             | 0.0302             | 0.0378             |
|                         | SLMRec   | _0.0443_ | _0.0651_ | _0.1038_ | _0.0249_ | _0.0303_ | _0.0382_ |
|                         | ADDVAE   | **0.0451**    | **0.0665**    | **0.1066**    | **0.0253**    | **0.0308**    | **0.0390**    |
|                         | FREEDOM  | 0.0396             | 0.0601             | 0.0998             | 0.0220             | 0.0273             | 0.0353             |

### Ablation Study

#### Recommendation performance comparison using different data split methods.:

We evaluate the performance of various recommendation models using different data splitting methods. The offline evaluation is based on the historical item ratings or the implicit item feedback. As this method relies on the user-item interactions and the models are all learning based on the supervised signals, we need to split the interactions into train, validation and test sets. There are three main split strategies that we applied to compare the performance:

• Random split: As the name suggested, this split strategy randomly selects the train and test boundary for each user, which selects to split the interactions according to the ratio. The disadvantage of the random splitting strategy is that they are not capable to reproduce unless the authors publish how the data split and this is not a realistic scenario without considering the time.

• User time split: The temporal split strategy splits the historical interactions based on the interaction timestamp by the ratio (e.g., train:validation:test=8:1:1). It split the last percentage of interactions the user made as the test set. Although it considers the timestamp, it is still not a realistic scenario because it is still splitting the train/test sets among all the interactions one user made but did not consider the global time.

• Global time split: The global time splitting strategy fixed the time point shared by all users according to the splitting ratio. The interactions after the last time point are split as the test set. Additionally, the users of the interactions after the global temporal boundary must be in the training set, which follows the most realistic and strict settings. The limitation of this strategy is that the number of users will be reduced due to the reason that the users not existing in the training set will be deleted

Our experiments on the Sports dataset, using these three splitting strategies, provide insights into their impact on recommendation performance. The table below presents the performance comparison results in terms of Recall@k and NDCG@k where k=10,20, and the second table shows the performance ranking of models based on Recall@20 and NDCG@20.

| Dataset | Model    |          | Recall@10 |             |          | Recall@20 |             |
|---------|----------|----------|-----------|-------------|----------|-----------|-------------|
|         |          | Random   | User Time | Global Time | Random   | User Time | Global Time |
|         | MMGCN    | 0.0384   | 0.0266    | 0.0140      | 0.0611   | 0.0446    | 0.0245      |
|         | BPR      | 0.0444   | 0.0322    | 0.0152      | 0.0663   | 0.0509    | 0.0258      |
|         | VBPR     | 0.0563   | 0.0385    | 0.0176      | 0.0851   | 0.0620    | 0.0298      |
|         | DualGNN  | 0.0576   | 0.0403    | 0.0181      | 0.0859   | 0.0611    | 0.0297      |
| sports  | GRCN     | 0.0604   | 0.0418    | 0.0167      | 0.0915   | 0.0666    | 0.0286      |
|         | LightGCN | 0.0568   | 0.0405    | 0.0205      | 0.0863   | 0.0663    | 0.0336      |
|         | LATTICE  | 0.0641   | 0.0450    | 0.0207      | 0.0964   | 0.0699    | 0.0337      |
|         | BM3      | 0.0646   | 0.0447    | 0.0213      | 0.0955   | 0.0724    | 0.0336      |
|         | SLMRec   | 0.0651   | 0.0470    | 0.0220      | 0.0985   | 0.0733    | 0.0350      |
|         | FREEDOM  | 0.0708   | 0.0490    | 0.0226      | 0.1080   | 0.0782    | 0.0372      |
| Dataset | Model    |          | NDCG@10   |             |          | NDCG@20   |             |
|         |          | Random   | User Time | Global Time | Random   | User Time | Global Time |
|         | MMGCN    | 0.0202   | 0.0134    | 0.0091      | 0.0261   | 0.0180    | 0.0125      |
|         | BPR      | 0.0245   | 0.0169    | 0.0102      | 0.0302   | 0.0218    | 0.0135      |
|         | VBPR     | 0.0304   | 0.0204    | 0.0115      | 0.0378   | 0.0265    | 0.0153      |
|         | DualGNN  | 0.0321   | 0.0214    | 0.0118      | 0.0394   | 0.0268    | 0.0155      |
| sports  | GRCN     | 0.0332   | 0.0219    | 0.0101      | 0.0412   | 0.0282    | 0.0138      |
|         | LightGCN | 0.0315   | 0.0220    | 0.0139      | 0.0391   | 0.0286    | 0.0180      |
|         | LATTICE  | 0.0351   | 0.0238    | 0.0138      | 0.0434   | 0.0302    | 0.0177      |
|         | BM3      | 0.0356   | 0.0237    | 0.0144      | 0.0436   | 0.0308    | 0.0182      |
|         | SLMRec   | 0.0364   | 0.0253    | 0.0148      | 0.0450   | 0.0321    | 0.0189      |
|         | FREEDOM  | 0.0388   | 0.0255    | 0.0151      | 0.0485   | 0.0330    | 0.0197      |

As demonstrated above, different data splitting strategies lead to varied performance outcomes for the same dataset and evaluation metrics. This variability presents a challenge in comparing the effectiveness of different models when they are based on different data split strategies.

|  Model   |        | Sports, NDCG@20   |             |
|----------|--------|-------------------|-------------|
|          | Random | User Time         | Global Time |
| MMGCN    | 10     | 10                | 10          |
| BPR      | 9      | 9                 | 8↑1         |
| VBPR     | 8      | 8                 | 7↑1         |
| LightGCN | 7      | 5↑2               | 4↑3         |
| DualGNN  | 6      | 7↓1               | 6           |
| DRCN     | 5      | 6↓1               | 9↓4         |
| LATTICE  | 4      | 4                 | 5↓1         |
| BM3      | 3      | 3                 | 3           |
| SLMRec   | 2      | 2                 | 2           |
| FREEDOM  | 1      | 1                 | 1           |
| **Model**    |        | **Sports, Recall@20** |             |
|          | Random | User Time         | Global Time |
| MMGCN    | 10     | 10                | 10          |
| BPR      | 9      | 9                 | 9           |
| VBPR     | 8      | 7↑1               | 6↑2         |
| DualGNN  | 7      | 8↓1               | 7           |
| LightGCN | 6      | 6                 | 5↑1         |
| GRCN     | 5      | 5                 | 8↓3         |
| BM3      | 4      | 3↑1               | 4           |
| LATTICE  | 3      | 4↓1               | 3           |
| SLMRec   | 2      | 2                 | 2           |
| FREEDOM  | 1      | 1                 | 1           |

The above table reports the ranks of SOTA models under each splitting strategy. The rows are sorted by the performance of models under random splitting strategy, with the up and down arrows indicating the relative rank position swaps compared with random splitting. As we can see, the ranking swaps are observed between the models under different splitting strategies

#### Recommendation performance comparison using Different Modalities
We are interested in how the modality information benefits the recommendation, and which modality contributes more. We aim to understand the specific benefits of different modalities in recommendation systems and provide guidelines for researchers on selecting appropriate modalities. We evaluate it by feeding the single modality information, and compare the performance between using both modalities and the single modality. 

The following figure is based on Recall@20 to show the summary and tendency of other modalities, visually summarize the impact of different modalities on various models. The orange point represents the performance of multi-modality, the green one represents the performance of textual modality and the blue point is for visual modality. The specific numerical values will be shown in our github.


<img src="https://github.com/hongyurain/Recommendation-with-modality-information/blob/main/IMG/modality-baby.jpg" alt="image-1" height="50%" width="50%" /><img src="https://github.com/hongyurain/Recommendation-with-modality-information/blob/main/IMG/modality-sports.jpg" alt="image-2" height="50%" width="50%" />

## Please consider to cite our paper those results helps you, thanks:
```

