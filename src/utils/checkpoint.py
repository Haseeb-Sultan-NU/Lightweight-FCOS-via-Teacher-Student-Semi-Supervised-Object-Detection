import os
import torch
import shutil

class CheckpointManager:
    def __init__(self, drive_save_path):
        """
        Args:
            drive_save_path (str): The path to your Google Drive checkpoints folder.
        """
        self.save_path = drive_save_path
        # Ensure the Drive directory actually exists
        os.makedirs(self.save_path, exist_ok=True)
        
    def save_checkpoint(self, epoch, model, optimizer, lr_scheduler, current_mAP, is_best=False):
        """
        Bulletproof save mechanism. Saves to a .tmp file first to prevent 
        corruption if Google Colab suddenly disconnects during the write process.
        """
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
            'best_mAP': current_mAP
        }
        
        # File paths
        temp_path = os.path.join(self.save_path, 'checkpoint_temp.pth')
        final_path = os.path.join(self.save_path, 'last_checkpoint.pth')
        best_path = os.path.join(self.save_path, 'best_oracle.pth')
        
        try:
            # 1. Save to the temporary file first (The safe way)
            torch.save(state, temp_path)
            
            # 2. Rename it to the actual file (Atomic operation, virtually instant)
            os.replace(temp_path, final_path)
            print(f"--> Checkpoint safely secured at Epoch {epoch} to Drive.")
            
            # 3. If this is the best model so far, make a dedicated copy
            if is_best:
                shutil.copyfile(final_path, best_path)
                print(f"--> New Best Model saved! (mAP: {current_mAP:.4f})")
                
        except Exception as e:
            print(f"CRITICAL WARNING: Failed to save checkpoint to Drive! Error: {e}")

    def load_checkpoint(self, checkpoint_path, model, optimizer=None, lr_scheduler=None):
        """
        Resumes training seamlessly.
        """
        if not os.path.isfile(checkpoint_path):
            print(f"No checkpoint found at '{checkpoint_path}'. Starting from scratch.")
            return 0, 0.0 # Returns starting epoch and starting mAP
            
        print(f"Loading checkpoint from '{checkpoint_path}'...")
        
        # Map to CPU first to avoid VRAM spikes, then move to GPU in the main loop
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load the architecture weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load the momentum and learning rates
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if lr_scheduler and checkpoint.get('scheduler_state_dict'):
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        best_mAP = checkpoint.get('best_mAP', 0.0)
        
        print(f"Successfully resumed from Epoch {checkpoint['epoch']} (Best mAP: {best_mAP:.4f})")
        return start_epoch, best_mAP