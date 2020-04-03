df2 <- read.csv(file="JData_Action_201604_unique.csv")
#write.csv(df2,file="JData_Action_201604_unique.csv",row.names=F)
df2$date <- as.Date(df2$time)
df2$time <- NULL
df2$model_id <- NULL
df2$brand <- NULL

# cate==8
df3 <- subset(df2,df2$cate==8 & df2$type != 4)
df3$cate <- NULL

# 04-11 - 04-15
df3_2 <- subset(df3,df3$date >"2016-04-10")

#user action count user table
df3_3 <- df3_2
df3_3$date <- NULL
df3_3 <- df3_3[order(df3_3$user_id),]
act_count <- data.frame(table(df3_3$user_id))
colnames(act_count) <- c("user_id","activity")
act_count <- act_count[order(-act_count$activity),]

# extract user_id with type==4 of month 2 and 3
df_m3 <- read.csv(file="JData_Action_201603_unique.csv")
df_m3t4 <- subset(df_m3,df_m3$cate==8 & df_m3$type==4)
df_m3t4 <- df_m3t4[c("user_id")]
rm(df_m3)

df_m2 <- read.csv(file="JData_Action_201602_unique.csv")
df_m2t4 <- subset(df_m2,df_m2$cate==8 & df_m2$type==4)
df_m2t4 <- df_m2t4[c("user_id")]
rm(df_m2)

#delete the users who have bought
#combine month2 and month3 result
df_m23t4 <- union(df_m2t4$user_id,df_m3t4$user_id)
act_count <- subset(act_count,!(act_count$user_id %in% df_m23t4))

# extract the top 500 active users
act_count$user_id <- as.numeric(as.character(act_count$user_id))
df3_users <- act_count[1:500,1]
df3_4 <- subset(df3_3,df3_3$user_id %in% df3_users)

df3_4t2 <- subset(df3_4,df3_4$type==2)
df3_4t3 <- subset(df3_4,df3_4$type==3)
df3_4usert23 <- intersect(df3_4t2$user_id,df3_4t3$user_id)
df3_4t2 <- subset(df3_4t2,df3_4t2$user_id %in% df3_4usert23)
df3_4t3 <- subset(df3_4t3,df3_4t3$user_id %in% df3_4usert23)

df3_4t2 <- transform(df3_4t2,usersku_id=paste0(user_id,sku_id))
df3_4t3 <- transform(df3_4t3,usersku_id=paste0(user_id,sku_id))
t2_count <- data.frame(table(df3_4t2$usersku_id))
colnames(t2_count) <- c("usersku","t2")
t3_count <- data.frame(table(df3_4t3$usersku_id))
colnames(t3_count) <- c("usersku","t3")
count_t23 <- merge(t3_count,t2_count,by="usersku")
# use t2 > t3 bug, extract t2 < =t3 may have it in the basket before
t23_delete <- subset(count_t23,count_t23$t2 <= count_t23$t3)
t23_delete$usersku <- as.numeric(as.character(t23_delete$usersku))
#delete the t2 <= t3 ones
df3_5 <- transform(df3_4,usersku_id=paste0(user_id,sku_id))
df3_5$usersku_id <- as.numeric(as.character(df3_5$usersku_id))
df3_5 <- subset(df3_5,!(df3_5$usersku_id %in% t23_delete$usersku))

# extract who added in the basket t2
df3_6 <- subset(df3_5,df3_5$type==2)
user_sku_final <- data.frame(unique(df3_6$usersku_id))
colnames(user_sku_final) <- c("user_sku_id")
user_sku_final$user_sku_id <- as.character(user_sku_final$user_sku_id)

user_sku_final$user_id <- as.numeric(substr(user_sku_final$user_sku_id,1,6))
user_sku_final$sku_id <- as.numeric(substr(user_sku_final$user_sku_id,7,12))

# which product will the user buy
# brand 
# attach brands and comment columns to the final user_sku list
pro <- read.csv(file="JData_Product.csv")
comm <- read.csv(file="JData_Comment.csv")
subcomm <- subset(comm,comm$dt=="2016-04-15")

user_sku_final <- merge(x=user_sku_final,y=pro[c("sku_id","brand")],by="sku_id",all.x =T)
user_sku_final <- merge(x=user_sku_final,y=subcomm[c("sku_id","comment_num","has_bad_comment","bad_comment_rate")],by="sku_id",all.x=T)
user_sku_final[is.na(user_sku_final)] <- -1
colnames(user_sku_final)[5:7] <- c("com_num","bad_comm","badcomm_rate")
# historical buying behavior users who have ever bought in the last three months
# use df_m3 and df_m2 and df2(for month4)
df_m3t4_2 <- subset(df_m3,df_m3$cate==8 & df_m3$type==4)
df_m2t4_2 <- subset(df_m2,df_m2$cate==8 & df_m2$type==4)
df_m23t4_2 <- rbind(df_m3t4_2,df_m2t4_2)
df_m4t4 <- subset(df2,df2$cate==8 & df2$type==4)
df_m234t4 <- rbind(df_m4t4,df_m23t4_2)
df_m234t4$date <- as.Date(df_m234t4$time)
# count the duplicated rows where sku_id and date are the same
abnormal <- df_m234t4[c("user_id","date")]
library(plyr)
abnormal <- ddply(abnormal,.(user_id,date),nrow)
colnames(abnormal)[3] <- "times"
# there's some seem to be abnormal ones
# top n popular brands in the last three months
popbrands <- data.frame(table(df_m234t4$brand))
colnames(popbrands) <- c("brand","times")
popbrands <- popbrands[order(-popbrands$times),]
#abnormal brands--- a test
abnormal2 <- df_m234t4[c("brand","date")]
abnormal2 <- ddply(abnormal2,.(brand,date),nrow)
colnames(abnormal2)[3] <- "times"
# users buying same products or brands on the sameday or different day
user_behavior <- df_m234t4[c("user_id","brand","date")]
user_behavior <- ddply(user_behavior,.(user_id,brand,date),nrow)
colnames(user_behavior)[4] <- "times"
table(user_behavior[4])
user_behavior <- ddply(user_behavior,.(user_id,brand),nrow)
colnames(user_behavior)[3] <- "times"
table(user_behavior[3])
user_behavior <- ddply(user_behavior,.(user_id,date),nrow)
colnames(user_behavior)[3] <- "times"
table(user_behavior[3])
# conclusion: users tend to buy on the same day and most buy same brand
# extract the user information from the final result
users_check <- ddply(user_sku_final,.(user_id),nrow)
colnames(users_check)[2] <- c("times")
table(users_check[2])
# users and products which have unique correspondence
sub_users1 <- subset(user_final,user_final$times==1)
sub_users1 <- sub_users1[['user_id']]
users_final <- subset(user_sku_final,user_sku_final$user_id %in% sub_users1)
final1 <- users_final[c("user_id","sku_id")]
# others
user_sku_final <- subset(user_sku_final,!(user_sku_final$user_id %in% final1$user_id))
popbrands$brand <- as.numeric(as.character(popbrands$brand))
# delete row where brands are not in top 10 purchase
final2 <- subset(user_sku_final,user_sku_final$brand %in% popbrands$brand[1:8])
final2 <- final2[c("sku_id","user_id")]
# select randomly
users_final2 <- ddply(final2,.(user_id),nrow)
colnames(users_final2)[2] <- "times"
tempsku <- NULL
for (i in 1:nrow(users_final2)) {
  temp_sub <- subset(final2,final2[2]==users_final2[i,1])
  tempsku <- append(tempsku,sample(temp_sub$sku_id,1),after=length(tempsku))
  rm(temp_sub)
}

final2 <- cbind(users_final2,tempsku)
final2[2] <- NULL
colnames(final2)[2] <- "sku_id"

final <- rbind(final1,final2)
final <- final[order(final[1]),]
rownames(final) <- 1:nrow(final)
write.csv(final,file="result.csv",row.names=F)
