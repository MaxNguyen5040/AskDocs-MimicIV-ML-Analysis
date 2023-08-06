# Notes: Any posts/comments are removed that are removed or deleted. If the entire post is deleted all comments are too, but if the comments are deleted the rest of the post remains intact.
# All comments from layman can be toggled to be removed or not.
# RC are comments, RS are posts


import json
import time
import gzip
import shutil
import os
start_time = time.time()

# ________Code to convert Gzip files into json_______
for subdir, dirs, files in os.walk("askdocs_input"):
    for file in files:
        print(file)

        with gzip.open("askdocs_input/"+file, 'rb') as file_in:
            with open("askdocs_output/"+file, 'wb') as file_out:
                shutil.copyfileobj(file_in, file_out)
                print('example.json file created')

def process_data(input_file):
    with open("askdocs_output/"+ input_file) as f:
        json_post_data = [json.loads(line) for line in f]

    post_data = {}
    for item in json_post_data:
        post_data.update({item["id"]: {"post_id": item["id"],"post_title":item["title"], "post_text": item["selftext"],"post_time_written": item["author_created_utc"],"post_upvotes": item["score"],"post_upvote_ratio": item["upvote_ratio"], "post_award_count": item["total_awards_received"], "post_link": item["permalink"]}})

    with open("askdocs_output/" + "RC"+input_file[2:]) as f:
        print("RC"+input_file[2:])
        json_comment_data = [json.loads(line) for line in f]

    comment_data  = {}
    for item in json_comment_data:
        if item["author"] != "AutoModerator":
            previous_list = comment_data.get(item["link_id"][3:],"no prior entries")
            if previous_list == "no prior entries": 
                comment_data.update({item["link_id"][3:]: [{"parent_id": item["link_id"][3:],"comment_id": item["id"],"comment_text": item["body"], "comment_flair": item["author_flair_text"],"comment_controversiality": item["controversiality"],"comment_award_count": item["total_awards_received"],"comment_upvotes": item["score"],"comment_time_written": item["author_created_utc"], "comment_link": item["permalink"]}]})
            else:
                previous_list.append({"parent_id": item["link_id"][3:],"comment_id": item["id"],"comment_text": item["body"], "comment_flair": item["author_flair_text"],"comment_controversiality": item["controversiality"],"comment_award_count": item["total_awards_received"],"comment_upvotes": item["score"],"comment_time_written": item["author_created_utc"], "comment_link": item["permalink"]})
                comment_data.update({item["link_id"][3:]: previous_list})

    post_comment_pairs = {}

    for id in list(post_data.keys()):
        if id in comment_data:
            comment_list = comment_data[id]
            if len(comment_list) > 1:
                new_comment_list = []
                for comment in comment_list:
                    if post_data[id]["post_text"] not in {"[deleted]","[removed]"}:
                        # if comment["comment_flair"] not in {"Layperson/not verified as healthcare professional.","Layperson/not verified as healthcare professional","This user has not yet been verified.","None"}:
                            if comment["comment_text"] not in {"[deleted]","[removed]"}:
                                new_comment_list.append(comment)            
                if len(new_comment_list) > 0:
                    post_comment_pairs.update({id:{"post": post_data[id], "list_of_comments": new_comment_list}})

            else:
                if not post_data[id]["post_text"] in {"[deleted]","[removed]"}:
                    #if not comment_list[0]["comment_flair"] in {"Layperson/not verified as healthcare professional.","Layperson/not verified as healthcare professional","This user has not yet been verified.","None"}:
                        if not comment_list[0]["comment_text"] in {"[deleted]","[removed]"}:
                            post_comment_pairs.update({id:{"post": post_data[id], "list_of_comments": [comment_list[0]]}})
        
    # for i in list(post_comment_pairs.items())[0:5]:
    #     print("\n----------")
    #     print('Post title: "' + i[1]['post']["post_title"]+'"')
    #     print("Post upvotes: " + str(i[1]['post']["post_upvotes"]))
    #     print('Post text: "' + i[1]['post']["post_text"]+'"')
    #     print("\n")
        
    #     count = 1
    #     for j in i[1]['list_of_comments']:
    #         print("__Comment #" + str(count))
    #         print("Upvotes: "+ str(j["comment_upvotes"]))
    #         print('Comment text: "'+ j["comment_text"]+'"\n')
    #         count +=1

    keyword = 0

    for i in list(post_comment_pairs.items()):
        if "pain" in i[1]['post']["post_text"]:
            keyword += 1
        
        for j in i[1]['list_of_comments']:
            if "pain" in j["comment_text"]:
                keyword += 1

    return len(post_comment_pairs),len(json_comment_data),len(json_post_data),keyword



# for subdir, dirs, files in os.walk("askdocs_output"):
#     num_comments = 0
#     num_posts = 0
#     num_pairs = 0
#     keyword = 0

#     for file in files:
#         if file[0:2] == "RS":
#             print(file)
#             a,b,c,d = process_data(file)
#             num_comments += b
#             num_posts += c
#             num_pairs += a
#             keyword += d
#     print(num_comments)
#     print(num_posts)
#     print(num_pairs)
#     print(keyword)
#     #695218
#     #278381
#     #53932
#     #3412

print("--- %s seconds ---" % (time.time() - start_time))
