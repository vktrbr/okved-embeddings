# Данные 
1. **data/graph/okved_graph.pickle** - граф для кодов ОКВЭД.
   *Как он создан? Вроде бы это граф связей между компаниями. Что на ребрах?*
   1. Type - id связи в графе
      1. 0 - связь компания-компания
      2. 1 - связь в классификаторе
      3. 2 - связь между разными кодами внутри одной компании
   2. Weight - ?
   3. Norm - ?? нормированный Weight. [Ссылка на документацию](https://docs.dgl.ai/en/latest/generated/dgl.nn.pytorch.conv.EdgeWeightNorm.html)
    

2. **data/okved2/okved_2014_w_sections.csv** - данные по кодам ОКВЭД.

# Модели