def get_hardcoded_block_coords(CHUNK_X,CHUNK_Z):

    block_x = (CHUNK_X * 16) + 9
    block_z = (CHUNK_Z * 16) + 9
    
    return block_x, block_z

a = [[-29, 29],[-81, 58],[-83, 60],[-71, -19],]

for i in a:
    final_x, final_z = get_hardcoded_block_coords(i[0], i[1])
    print(final_x,final_z)
