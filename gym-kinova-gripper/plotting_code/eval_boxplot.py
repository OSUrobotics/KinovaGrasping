import pandas as pd

def create_boxplot(saving_dir,data,labels,filename):
    """ Create a boxplot to show averages and spread of data
    saving_dir: Directory where plot will be saved
    x: x-axis data
    y: y-axis data
    labels: dictionary of plot labels in string format {"x_label":,"y_label":,"title":}
    """
    print("In create_boxplot, data: ",data)
    dataframe = pd.DataFrame(data)
    print("In create_boxplot, data: ", dataframe)
    boxplot = sns.boxplot(data=data)
    print("post sns.boxplot call")
    boxplot.savefig(saving_dir+filename)
    print("post savefig")
    print("**Boxplot created at: ",saving_dir+filename)


"""
# Plot evaluation boxplot to see reward distribution
episode_reward_values = eval_ret["total_avg_reward_values"]

if episode_num == int(args.eval_freq):
    eval_values = np.array([episode_num])
    all_boxplot_data = episode_reward_values
    print("A) eval_values: ",eval_values)
else:
    eval_values = np.arange(args.eval_freq,episode_num,args.eval_freq) # start, stop, step
    all_boxplot_data = np.append(all_boxplot_data, episode_reward_values)
    print("B) eval_values: ", eval_values)

#boxplot_data = [eval_values, episode_reward_values]
boxplot_data = {"Evaluation Episode":eval_values,"Total Avg. Reward":all_boxplot_data}
print("boxplot_data: ",boxplot_data)
boxplot_labels = {"x_label": "Evaluation Episode", "y_label": "Total Avg. Reward",
          "title": "Total Avg. Reward per " + str(eval_episodes)}
create_boxplot(evplot_saving_dir,boxplot_data,boxplot_labels,"Eval_Boxplot.png")
print("Post create_boxplot")
"""