from matplotlib import pyplot as plt
from datetime import datetime
import os
import sys
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def plot_one_relative_depth_img(relative_distance_dict, relative_depth_img,gt_frame_id):
    plt.close()
    plt.figure(figsize=(10, 6))
    cm_depth = 'jet' # 'jet' 'coolwarm'
    min_value = min(relative_distance_dict.values())-0.2 
    plt.imshow(relative_depth_img, cmap=cm_depth, vmin=min_value)
    plt.title(f'gt relative_depth_image (gt Frame ID: {gt_frame_id})') 
    plt.colorbar()  # Display a colorbar for the depth scale
    plt.axis('off')
    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.05)  # You can adjust the value as needed
    plt.show()

def plot_2_depth_img(gt_relative_depth_img,est_relative_depth_img,depth_frame_ts,mask_timestamp,sequence,show = 0,save = 0):

    # min_value = min(np.min(gt_relative_depth_img), np.min(est_relative_depth_img))
    # max_value = max(np.max(gt_relative_depth_img), np.max(est_relative_depth_img))
    min_value = np.min(gt_relative_depth_img)
    max_value = np.max(gt_relative_depth_img)
    # print("Minimum Value:", min_value, "Maximum Value:", max_value)
    min_value = min_value -0.2 

    # plt.close() 

    plt.figure(figsize=(20, 6))
    cm_depth =  'jet'# 'viridis' # 'jet' 'coolwarm'
    plt.subplot(1, 2, 1)
    plt.imshow(est_relative_depth_img, cmap=cm_depth, vmin=min_value, vmax=max_value)
    plt.title(f'est relative_depth_image (depth_frame_ts: {depth_frame_ts})') 
    plt.colorbar()  # Display a colorbar for the depth scale
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(gt_relative_depth_img, cmap=cm_depth, vmin=min_value, vmax=max_value)
    plt.title(f'gt relative_depth_image (gt timestamp: {mask_timestamp})') 
    plt.colorbar()  # Display a colorbar for the depth scale
    plt.axis('off')
    if show:
        # plt.show()
        pass
    if save:
        save_dir = 'output'
        current_date = datetime.now().strftime('%Y-%m-%d')
        save_path = os.path.join(save_dir, current_date + 'stppp_gt_plt', sequence)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, str(mask_timestamp)+ '.png'), bbox_inches='tight')
        print('savefig',os.path.join(save_path, str(mask_timestamp)+ '.png'))
        sys.stdout.flush()


def plot_rot_flows(mask,rot_flow_mat_dic,gt_frame_id,est_relative_distance_dict,est_relative_depth_img,t,sequence,save = 0):
    mask_height, mask_width = mask.shape
    plt.close() 
    plt.figure(figsize=(mask_height * 0.02 *2, mask_width * 0.018)) 
    plt.subplot(1, 2, 1)
    colors = ['xkcd:light grey',
            'xkcd:salmon',
            'xkcd:plum',
            'xkcd:lightblue',
            'xkcd:gold',
            'xkcd:darkblue',
            'xkcd:white']
    cm = LinearSegmentedColormap.from_list('foo', colors[0:29], 29)
    im = plt.imshow(mask, cmap=cm)
    # plt.title(f'Mask Image (est_frame_id= {est_frame_id})') 
    # plt.colorbar(im,shrink=0.75)
    plt.clim(0, 29000)
    plt.tight_layout()

    for object_id in rot_flow_mat_dic:
        rot_flow = rot_flow_mat_dic[object_id]
        rot_flow = rot_flow * 20
        mass_y, mass_x  = np.where(mask == object_id)  # this is right order of y,x
        cent_x = np.mean(mass_x)
        cent_y = np.mean(mass_y)   
        relative_distance = est_relative_distance_dict[object_id]
        # text = f'ob{int(object_id/1000)} relDis {relative_distance:.2f}\noptFlow [{rot_opt_flow_u:.2f},{rot_opt_flow_v:.2f}]'
        # print('rot_flow_mat_dic',rot_flow_mat_dic)
        text = f'ob{int(object_id/1000)} relDis {relative_distance:.2f}\noptFlow [{rot_flow[0,0]:.2f},{rot_flow[1,0]:.2f}]'
        color = 'red' if relative_distance < 0 or relative_distance >2000 else 'lemonchiffon'
        plt.text(cent_x, cent_y, text, color='red'if relative_distance <= 0 else 'b', bbox=dict(fc='yellow', alpha=0.5) if relative_distance <= 0 or relative_distance >2000 else None)
        # plt.arrow(cent_x, cent_y, A*100, B*100, color=color, head_width=5, head_length=5)  # aligned rotation velocity
        plt.arrow(cent_x, cent_y, rot_flow[0,0], rot_flow[1,0], color=color, head_width=5, head_length=5)   # flow vec
    
    plt.title(f'Mask Image (gt_frame_id= {gt_frame_id:.3f})') 
    plt.axis('off')
    plt.colorbar(im,shrink=0.75)

    plt.subplot(1, 2, 2)
    # est_relative_obj_distance_img = np.zeros_like(mask)
    # for object_id in relative_obj_distance_dict:
    #     est_relative_obj_distance_img = np.where(m == object_id,relative_obj_distance_dict[object_id],est_relative_obj_distance_img)
    # plt.title('Depth Image')
    plt.title(f'est relative_depth_image (t: {t})') 
    min_value = np.min(est_relative_depth_img) -0.1
    max_value = np.max(est_relative_depth_img) +0.1

    im_depth = plt.imshow(est_relative_depth_img, cmap='jet', vmin=min_value, vmax=max_value)
    plt.axis('off')
    plt.colorbar(im_depth)
    if save:
        print('save')

        save_dir = 'output'
        current_date = datetime.now().strftime('%Y-%m-%d')
        save_path = os.path.join(save_dir, current_date + 'plot_rot_flows')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file_name = os.path.join(save_path,str(sequence)+ '_'+str(t)+ '.png')
        plt.savefig(save_file_name, bbox_inches='tight')
        print('save_file_name',save_file_name)
        sys.stdout.flush()

    else:
        plt.show()

def plot_mask_depth(mask_frame_id,mask,depth_frame_ts,depth,plot_FLAG = 0):  
    if plot_FLAG:
        
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 2, 1)
        plt.title(f'Mask Image (Frame ID: {mask_frame_id})') 
        colors = ['xkcd:light grey',
                'xkcd:salmon',
                'xkcd:plum',
                'xkcd:lightblue',
                'xkcd:gold',
                'xkcd:darkblue',
                'xkcd:white']
        cm = LinearSegmentedColormap.from_list('foo', colors[0:50], 50)
        im_mask = plt.imshow(mask, cmap=cm)
        plt.clim(0, 7200)
        plt.axis('off')
        plt.colorbar(im_mask)

        # Plot depth image in the second subplot
        plt.subplot(1, 2, 2)
        # plt.title('Depth Image')
        plt.title(f'Depth Image ts: {depth_frame_ts}')
        # im_depth = plt.imshow(depth, cmap='gray')
        im_depth = plt.imshow(depth, cmap='viridis') #, vmin=200, vmax=900)
        # im_depth = plt.imshow(depth, cmap='viridis')
        plt.axis('off')
        plt.colorbar(im_depth)
        plt.tight_layout()
        # plt.show()    
        # mngr = plt.get_current_fig_manager()
        # # to put it into the upper left corner for example:
        # mngr.window.setGeometry(50,100,640, 545)

def e2depth_plot(est_average_depth_img,depth_frame_id,avg_obj_depth_dict,show = 0):
    if show:
        plt.close()
        # some function is the same as relative_dis
        plt.figure(figsize=(10, 6))
        cm_depth = 'viridis' # 'jet' 'coolwarm'
        # min_value = min(avg_obj_depth_dict.values())  
        # max_value = max(avg_obj_depth_dict.values())
        # print("avg_obj_depth_dict, Minimum Value:", min_value, "Maximum Value:", max_value)  # sometimes only have one key (one object have depth)
        # # continue
        # min_value = min_value - 0.5 
        # max_value = max_value + 0.5
        # if max_value:
        #     plt.imshow(est_average_depth_img, cmap=cm_depth, vmin=min_value, vmax=max_value)
        # else:
        plt.imshow(est_average_depth_img, cmap=cm_depth)
        plt.title(f'est_average_depth_img (depth_frame_id: {depth_frame_id}) Object ID: {list(avg_obj_depth_dict.keys())}\nAverage Depth: {list(avg_obj_depth_dict.values())}') 
        plt.colorbar()  # Display a colorbar for the depth scale
        plt.axis('off')
        plt.show()

def overlay_mask_on_depthmap(mask_frame_id,mask,depth_frame_ts, depth_img,plot_FLAG = 0):
    if plot_FLAG:
        # Create a figure and a single subplot
        overlay_fig = plt.figure(figsize=(10, 5))
        plt.title(f'Mask Overlay on Depth Image (Frame ID: {mask_frame_id},depth frame ts: {depth_frame_ts})')

        colors = ['xkcd:light grey',
                'xkcd:salmon',
                'xkcd:plum',
                'xkcd:lightblue',
                'xkcd:gold',
                'xkcd:darkblue',
                'xkcd:white']
        cm = LinearSegmentedColormap.from_list('foo', colors[0:29], 29)

        plt.imshow(depth_img, cmap='gray')
        # plt.clim(0, 450)
        plt.axis('off')

        # Overlay the mask with transparency
        plt.imshow(mask, cmap=cm, alpha=0.5)
        # plt.colorbar()
        plt.show()

def plot_4_confidence(est_relative_depth_img,sigma3_img,depth,gt_relative_depth_img,mask,frame_id,frame_ts,sequence):
    # res distance   confidence
    # gt origin depth   gt rela depth
    plt.figure(figsize=(20, 12))
    cm_depth =  'jet'# 'viridis' # 'jet' 'coolwarm'

    max_value = np.max(gt_relative_depth_img)
    plt.subplot(2, 2, 1)
    plt.imshow(est_relative_depth_img, cmap=cm_depth, vmax=max_value)#, vmin=min_value, vmax=max_value)
    plt.title(f'est relative_depth_image frame id {frame_id} ')#(depth_frame_ts: {depth_frame_ts})') 
    plt.colorbar()  # Display a colorbar for the depth scale
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(sigma3_img, cmap='gray', vmax=0.6) #, vmin=min_value, vmax=max_value)
    plt.title(f'sigma3_img')#(depth_frame_ts: {depth_frame_ts})') 
    plt.colorbar()  # Display a colorbar for the depth scale
    plt.axis('off')



    plt.subplot(2, 2, 3)
    plt.imshow(gt_relative_depth_img, cmap=cm_depth,vmax=max_value)
    plt.title(f'gt relative_depth_image') 
    plt.colorbar()  # Display a colorbar for the depth scale
    plt.axis('off')


    plt.subplot(2, 2, 4)
    plt.imshow(depth, cmap=cm_depth)#, vmin=min_value, vmax=max_value)
    plt.title(f'origin depth')#(depth_frame_ts: {depth_frame_ts})') 
    plt.colorbar()  # Display a colorbar for the depth scale
    plt.axis('off')
    # colors = ['xkcd:light grey',
    #         'xkcd:salmon',
    #         'xkcd:plum',
    #         'xkcd:lightblue',
    #         'xkcd:gold',
    #         'xkcd:darkblue',
    #         'xkcd:white']
    # cm = LinearSegmentedColormap.from_list('foo', colors[0:29], 29)
    # plt.imshow(mask, cmap=cm, alpha=0.2)

    # plt.show()
    plt.tight_layout()

    save_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/stppp_and_output/x-student-events/ECCV_plot/output'
    current_date = datetime.now().strftime('%Y-%m-%d')
    save_path = os.path.join(save_dir, current_date + '_honey_plt')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_name = os.path.join(save_path,str(sequence)+ '_'+ str(frame_id)+'_'+str(frame_ts)+ '.png')
    print('save_file_name',save_file_name)
    plt.savefig(save_file_name, bbox_inches='tight',transparent = True)
    sys.stdout.flush()


def save_4_confidence(est_relative_depth_img,sigma3_img,depth,gt_relative_depth_img,mask,frame_id,frame_ts,sequence):

    plt.figure(figsize=(20, 12))
    cm_depth = 'jet'  # 'viridis' # 'jet' 'coolwarm'

    max_value = np.max(gt_relative_depth_img)

    save_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/stppp_and_output/x-student-events/ECCV_plot/honeyres'

    # Subplot 1
    plt.subplot(2, 2, 1)
    plt.imshow(est_relative_depth_img, cmap=cm_depth, vmax=max_value)
    # plt.title(f'est relative_depth_image frame id {frame_id}')
    # plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    file_name = 'ours_d'
    # plt.savefig('subplot1.png', transparent=True)  # Save the first subplot
    plt.savefig(f'{save_dir}/{sequence}_{frame_id}_{frame_ts:.3f}_{file_name}.png', bbox_inches='tight', transparent=True)



    # Subplot 3
    plt.figure(figsize=(20, 12))  # Create another new figure
    plt.subplot(2, 2, 3)
    plt.imshow(gt_relative_depth_img, cmap=cm_depth, vmax=max_value)
    # plt.title(f'gt relative_depth_image')
    # plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    file_name = 'hc_gt'
    # plt.savefig('subplot3.png', transparent=True)  # Save the third subplot
    plt.savefig(f'{save_dir}/{sequence}_{frame_id}_{frame_ts:.3f}_{file_name}.png', bbox_inches='tight', transparent=True)

    # Subplot 4
    plt.figure(figsize=(20, 12))  # Create a new figure
    plt.subplot(2, 2, 4)
    plt.imshow(depth, cmap=cm_depth, vmax=max_value)
    # plt.title(f'origin depth')
    # plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    file_name = 'orig_gt'
    # plt.savefig('subplot4.png', transparent=True)  
    plt.savefig(f'{save_dir}/{sequence}_{frame_id}_{frame_ts:.3f}_{file_name}.png', bbox_inches='tight', transparent=True)

    # Subplot 2
    plt.figure(figsize=(20, 12))  # Create a new figure
    plt.subplot(2, 2, 2)
    plt.imshow(sigma3_img, cmap='gray', vmax=0.2)
    # plt.title(f'sigma3_img')
    # plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    file_name = 'ours_sigma3'
    # plt.savefig('subplot2.png', transparent=True)  # Save the second subplot
    plt.savefig(f'{save_dir}/{sequence}_{frame_id}_{frame_ts:.3f}_{file_name}.png', bbox_inches='tight', transparent=True)
