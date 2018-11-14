library(dplyr)
df <- read.csv("../../../imgclsmob_data/oi4bb/validation-annotations-bbox.csv")
df2 <- df %>%
  mutate(Square = (XMax - XMin) * (YMax - YMin)) %>%
  select(ImageID, LabelName, Square) %>%
  group_by(ImageID) %>%
  slice(which.max(Square)) %>%
  select(ImageID, LabelName)
write.csv(df2, "../../../imgclsmob_data/oi4bb/validation-cls.csv", quote = FALSE, row.names = FALSE)
